# This module contains functions to construct the R tensor from the W tensor.

module ConstructRTensorGPU

using HDF5
using CUDA
using StaticArrays
using SpecialFunctions

include("../../utils/BinEncoding.jl")

using .BinEncoding: decode_bins
using ...FastPowers: fast_i_pow, fast_neg1_pow
using ...SparseTensors
using ...SparseTensors: SparseWTensor, SparseGauntArray

export construct_R_tensor_gpu

@inline function load_gaunt_array(path::String, ::Type{T}) where {T<:AbstractFloat}
    """
    Load a SparseGauntArray from an HDF5 file.

    This function reads the Gaunt coefficients and associated indices from a HDF5 file
    and reconstructs the SparseGauntArray structure. The bins are decoded from the
    flattened HDF5 representation.

    # Arguments:
    - path::String: Path to the HDF5 file containing the Gaunt coefficients.
    - T::Type: The floating point type to use for the coefficients.

    # Returns:
    - SparseGauntArray{T}: The loaded Gaunt coefficient array.
    """
    h5open(path, "r") do io
        # Read in the indices and coefficients.
        lambda = Vector{Int}(read(io, "lambda"))
        mu = Vector{Int}(read(io, "mu"))
        L = Vector{Int}(read(io, "L"))
        l = Vector{Int}(read(io, "l"))
        m = Vector{Int}(read(io, "m"))
        coeffs = Vector{T}(read(io, "coefficients"))
        # Reconstruct the lambda_mu_bins.
        lambda_mu_bins = decode_bins(Vector{Int}(read(io, "lambda_mu_bins/data")), Vector{Int}(read(io, "lambda_mu_bins/offsets")))
        return SparseGauntArray(lambda, mu, L, l, m, coeffs, lambda_mu_bins)
    end
end

# The threshold below which we set j_L(x) -> j_L(0) for stability.
# The largest uncertainty from this will be from j_1(x_min) - j_1(0) ≃ x_min/3.
const SMALL_X_THRESHOLD = 1.0e-3

const KEV_TO_INV_ANGSTROM = 1.0 / 1.973269804  # Multiplicative factor to convert keV to inverse Å.
const INV_FOUR_PI = 1.0 / (4.0 * π)
const prefactor = 2.0 * sqrt(2) * (2π)^(5 / 2)

const MAX_L_GLOBAL = 96 # The maximum L value we support globally for spherical Bessel functions.

@inline function compute_q_powers(
        q_grid::Vector{T},
        n_max::Int,
        n_q::Int,
    ) where {T<:AbstractFloat}
    """
    Compute powers of q from 0 to n_max on the CPU, then transfer to the GPU.

    # Arguments:
    - q_grid::Vector{T}: The grid of q values.
    - n_max::Int: The maximum power of q to compute.
    - n_q::Int: The number of q values in the q_grid.

    # Returns:
    - CuArray{T, 2}: An array of q powers, with dimensions (n_max + 1, n_q).
    """

    q_powers_cpu = Array{T, 2}(undef, n_max + 1, n_q)
    typed_one = one(T)

    @inbounds for q_idx in 1:n_q
        q_val = q_grid[q_idx]
        q_pow = typed_one
        q_powers_cpu[1, q_idx] = typed_one
        @inbounds for n in 1:n_max
            q_pow *= q_val
            q_powers_cpu[n + 1, q_idx] = q_pow
        end
    end

    return CuArray(q_powers_cpu)
end

function precompute_spherical_harmonic_normalisation(::Type{T}, L_max::Int) where {T<:AbstractFloat}
    """
    Precompute the normalisation constants for spherical harmonics Y_L^M, defined by:
        
        N_{L,M} = sqrt((2L + 1)/(4π) * (L - M)!/(L + M)!).
        
    These are only computed for M > 0, and L geq |M|. We store them with the linear key:

        key(L, M) = (L * (L + 1)) / 2 + M + 1.

    # Arguments:
    - L_max::Int: The maximum L value to compute.

    # Returns:
    - Array{T, 1}: An array of normalisation constants.
    """

    # Allocate the output array.
    n_entries = ((L_max + 1) * (L_max + 2)) ÷ 2
    Ylm_norms = Array{T, 1}(undef, n_entries)
    inv_four_pi = T(INV_FOUR_PI)

    @inbounds for L in 0:L_max
        # Precompute as much as possible.
        sqrt_term = sqrt((2*L + 1) * inv_four_pi)
        key_base = (L * (L + 1)) ÷ 2 + 1
        for M in 0:L
            key = key_base + M
            # The factorial part has to be computed with loggamma to avoid overflow.
            norm = sqrt_term * sqrt(exp(loggamma(L - M + 1) - loggamma(L + M + 1)))
            Ylm_norms[key] = norm
        end
    end

    return Ylm_norms
end

function sphercial_harmonics_kernel!(
        Ylm_tensor::CuDeviceArray{Complex{T}, 2},
        Ylm_norms::CuDeviceVector{T},
        L_max::Int,
        R_ij_hat::CuDeviceArray{T, 3},
        ij_i::CuDeviceVector{Int32},
        ij_j::CuDeviceVector{Int32}
    ) where {T<:AbstractFloat}
    """
    A kernel to compute the spherical harmonics Y_L^M(θ, ϕ) for all (i, j) pairs up to L_max.
    Each thread computes a unique (i, j) pair. The (L,M) spherical harmonics are stored with the
    linear key:

        key(L, M) = (L * (L + 1)) / 2 + M + 1.

    # Arguments:
    - Ylm_tensor::CuArray{Complex{T}, 2}: The output tensor to store the spherical harmonics.
    - L_max::Int: The maximum L value to compute.
    - R_ij_hat::CuArray{T, 3}: The (θ, ϕ) angles for the unit vectors Rhat_{ij}.
    - ij_i::CuArray{Int32}: The i indices for the non-zero (i, j) bins.
    - ij_j::CuArray{Int32}: The j indices for the non-zero (i, j) bins.
    """

    # Get the global thread index and stride.
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    n_bins = size(ij_i, 1)
    n_lm = ((L_max + 1) * (L_max + 2)) ÷ 2

    @inbounds while idx <= n_bins
        # Compute the (i, j) indices from the 1D index.
        idx0 = idx - 1
        i = ij_i[idx]
        j = ij_j[idx]
        theta = R_ij_hat[i, j, 1]
        phi = R_ij_hat[i, j, 2]
        st, ct = sincos(theta) # For our purposes, sqrt(1-x^2) = sin(theta).
        phase_factor = Complex{T}(cos(phi), sin(phi))

        # Compute the SHs and associated Legendre polynomials in one sweep.
        # Seed the recurrence. We will keep track of (M-1, L-1, L-2) etc. in temporary variables.
        P_MM_m1 = one(T)  # P_0^0 = 1.
        phase = one(Complex{T}) # Running phase factor, exp(i M ϕ).

        @inbounds for M in 0:L_max
            # Start with the diagonal terms.
            key = (M * (M + 1)) ÷ 2 + M + 1

            if M == 0
                P_MM = P_MM_m1
            else
                P_MM = -(2*M - 1) * st * P_MM_m1 # P_M^M(cos(θ)) = -(2M - 1) sin(θ) P_{M-1}^{M-1}(cos(θ)).
                P_MM_m1 = P_MM
                phase *= phase_factor
            end

            # Convert this to the correct spherical harmonic and store it.
            Ylm_tensor[idx, key] = Complex{T}(Ylm_norms[key] * P_MM, zero(T)) * phase

            # There are no terms with L > L_max.
            if M == L_max
                continue
            end

            # Now compute the off-diagonal terms.
            P_LM_m2 = P_MM
            P_LM_m1 = (2*M + 1) * ct * P_MM # P_{M+1}^M(cos(θ)) = (2M + 1) cos(θ) P_M^M(cos(θ)).

            # Store the L = M + 1 term.
            key = ((M + 1) * (M + 2)) ÷ 2 + M + 1
            Ylm_tensor[idx, key] = Complex{T}(Ylm_norms[key] * P_LM_m1, zero(T)) * phase

            # Now compute the remaining off diagonal terms.
            for L in (M + 2):L_max
                P_LM = ((2*L - 1) * ct * P_LM_m1 - (L + M - 1) * P_LM_m2) / (L - M)

                # Store the spherical harmonic.
                key = (L * (L + 1)) ÷ 2 + M + 1
                Ylm_tensor[idx, key] = Complex{T}(Ylm_norms[key] * P_LM, zero(T)) * phase

                # Update the temporary variables for the next iteration.
                P_LM_m2 = P_LM_m1
                P_LM_m1 = P_LM
            end
        end

        idx += stride
    end
end

function R_tensor_kernel!(
    R_pos_real::CuDeviceArray{T, 3},
    R_pos_imag::CuDeviceArray{T, 3},
    R_neg_real::CuDeviceArray{T, 3},
    R_neg_imag::CuDeviceArray{T, 3},
    Ylm_tensor::CuDeviceArray{Complex{T},2},
    weighted_bessel_tensor::CuDeviceArray{T,3},
    W_values::CuDeviceVector{Complex{T}},
    W_lambda::CuDeviceVector{Int32},
    W_mu::CuDeviceVector{Int32},
    W_n::CuDeviceVector{Int32},
    W_bin_offsets::CuDeviceVector{Int32},
    gaunt_coeffs::CuDeviceVector{T},
    gaunt_lambda_mu_indices::CuDeviceVector{Int32},
    gaunt_lambda_mu_offsets::CuDeviceVector{Int32},
    gaunt_mu::CuDeviceVector{Int32},
    gaunt_L::CuDeviceVector{Int32},
    gaunt_l::CuDeviceVector{Int32},
    gaunt_m::CuDeviceVector{Int32},
    q_powers::CuDeviceMatrix{T},
    i_powers::CuDeviceVector{Complex{T}},
    neg1_powers::CuDeviceVector{T},
    tdm_prefactors::CuDeviceMatrix{T},
    threshold_values::CuDeviceVector{T},
    n_q::Int32,
    n_bins::Int32,
    l_max::Int32,
    lambda_max::Int32,
    L_max_global::Int32,
    total::Int64,
) where {T<:AbstractFloat}
    """
    A kernel to construct the R tensor. Each thread processes all W tensor entries for one (transition, q, bin) triplet
    and atomically accumulates into R_{lm}. The mapping from the 1D thread index to (transition_idx, q_idx, bin_idx) is
    given by:

        bin_idx = idx % n_bins,
        q_idx = (idx ÷ n_bins) % n_q,
        transition_idx = idx ÷ (n_bins * n_q).

    For atomic adds, we have to deal with the real and imaginary parts separately.

    # Arguments:
    - R_pos_real::CuDeviceArray{T, 3}: The real part of the R tensor for m ≥ 0 stored on the GPU.
    - R_pos_imag::CuDeviceArray{T, 3}: The imaginary part of the R tensor for m ≥ 0 stored on the GPU.
    - R_neg_real::CuDeviceArray{T, 3}: The real part of the R tensor for m < 0 stored on the GPU.
    - R_neg_imag::CuDeviceArray{T, 3}: The imaginary part of the R tensor for m < 0 stored on the GPU.
    - Ylm_tensor::CuDeviceArray{Complex{T}, 2}: The precomputed spherical harmonics tensor.
    - weighted_bessel_tensor::CuDeviceArray{T, 3}: The weighted spherical Bessel functions tensor.
    - W_values::CuDeviceVector{Complex{T}}: The sparse W values stored on the GPU.
    - W_lambda::CuDeviceVector{Int32}: The λ indices stored on the GPU.
    - W_mu::CuDeviceVector{Int32}: The μ indices stored on the GPU.
    - W_n::CuDeviceVector{Int32}: The n indices stored on the GPU.
    - W_bin_offsets::CuDeviceVector{Int32}: Offsets into the W arrays for each (i, j) bin.
    - gaunt_coeffs::CuDeviceVector{T}: The Gaunt coefficients stored on the GPU.
    - gaunt_lambda_mu_indices::CuDeviceVector{Int32}: The (λ, μ) indices for Gaunt bins stored on the GPU.
    - gaunt_lambda_mu_offsets::CuDeviceVector{Int32}: The (λ, μ) offsets for Gaunt bins stored on the GPU.
    - gaunt_mu::CuDeviceVector{Int32}: The μ values for each Gaunt coefficient stored on the GPU.
    - gaunt_L::CuDeviceVector{Int32}: The L values for each Gaunt coefficient stored on the GPU.
    - gaunt_l::CuDeviceVector{Int32}: The ℓ values for each Gaunt coefficient stored on the GPU.
    - gaunt_m::CuDeviceVector{Int32}: The m values for each Gaunt coefficient stored on the GPU.
    - q_powers::CuDeviceMatrix{T}: The precomputed q powers stored on the GPU.
    - i_powers::CuDeviceVector{Complex{T}}: The precomputed powers of i (the imaginary unit).
    - neg1_powers::CuDeviceVector{T}: The precomputed powers of -1.
    - tdm_prefactors::CuDeviceMatrix{T}: The transition matrix prefactors for each (transition, bin) pair.
    - threshold_values::CuDeviceVector{T}: The per-transition threshold values for discarding small contributions.
    - n_q::Int32: The number of q points.
    - n_bins::Int32: The number of non-zero (i, j) bins.
    - l_max::Int32: The maximum ℓ value requested.
    - lambda_max::Int32: The maximum λ value for the molecule.
    - L_max_global::Int32: The maximum L value used for indexing neg1_powers.
    - total::Int64: The total number of (transition_idx, W_idx, q_idx) triplets to process.
    """

    # Get the global thread index and stride.
    # This is done in Int64 to avoid overflow for large total sizes.
    idx = Int64((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int64(gridDim().x * blockDim().x)

    @inbounds while idx <= total
        # Map the 1D thread index to (transition_idx, q_idx, bin_idx).
        idx0 = idx - 1
        bin_idx = Int32(mod(idx0, Int64(n_bins))) + 1
        tq_idx = idx0 ÷ Int64(n_bins)
        q_idx = Int32(mod(tq_idx, Int64(n_q))) + 1
        transition_idx = Int32(tq_idx ÷ Int64(n_q)) + 1

        # Determine the W range for this bin.
        W_start = W_bin_offsets[bin_idx]
        W_end = W_bin_offsets[bin_idx + Int32(1)] - Int32(1)

        TDM_prefactor_bin = tdm_prefactors[transition_idx, bin_idx]

        @inbounds for W_idx in W_start:W_end
            # Fetch the W tensor entry data for this thread.
            W_val = W_values[W_idx]
            W_n_val = W_n[W_idx]
            W_lambda_val = W_lambda[W_idx]
            W_mu_val = W_mu[W_idx]

            # Apply per-transition thresholding before looping over Gaunt coefficients.
            if abs(TDM_prefactor_bin) * abs(W_val) < threshold_values[transition_idx]
                continue
            end

            # Get the corresponding q^n value.
            q_n = q_powers[W_n_val + 1, q_idx]

            # Find the matching Gaunt coefficients for this (λ, μ) pair.
            bin_key = W_lambda_val * W_lambda_val + (W_lambda_val + W_mu_val) + 1
            gaunt_bin_start = gaunt_lambda_mu_offsets[bin_key]
            gaunt_bin_end = gaunt_lambda_mu_offsets[bin_key + 1] - 1

            # Skip if there are no Gaunt coefficients for this (λ, μ) pair.
            if gaunt_bin_end < gaunt_bin_start
                continue
            end

            # Loop over all Gaunt coefficients in this (λ, μ) bin.
            @inbounds for gaunt_position in gaunt_bin_start:gaunt_bin_end
                gaunt_idx = gaunt_lambda_mu_indices[gaunt_position]
                gaunt_L_val = gaunt_L[gaunt_idx]
                gaunt_l_val = gaunt_l[gaunt_idx]
                gaunt_m_val = gaunt_m[gaunt_idx]
                gaunt_coeff = gaunt_coeffs[gaunt_idx]

                # Skip if this Gaunt coefficient is outside the allowed range.
                # This should never happen in practice because of how we construct them, but it's here as a safeguard.
                if gaunt_l_val > l_max || abs(gaunt_m_val) > l_max
                    continue
                end

                # Compute M = m - μ for the spherical harmonic Y_L^M.
                M = gaunt_m_val - W_mu_val

                # If required, compute Y_L^{-M} using the symmetry relation.
                if M >= 0
                    key = (gaunt_L_val * (gaunt_L_val + 1)) ÷ 2 + M + 1
                    Y_LM_conj = conj(Ylm_tensor[key, bin_idx])
                else
                    key = (gaunt_L_val * (gaunt_L_val + 1)) ÷ 2 + (-M) + 1
                    Y_LM = Ylm_tensor[key, bin_idx]
                    Y_LM_conj = neg1_powers[M + L_max_global + 1] * Y_LM
                end

                # Compute the contribution to R_{lm} from this Gaunt coefficient and W entry.
                R_val = TDM_prefactor_bin * gaunt_coeff * q_n * W_val * weighted_bessel_tensor[gaunt_L_val + 1, bin_idx, q_idx] * i_powers[gaunt_L_val + 1] * Y_LM_conj

                # Atomically accumulate into R_{lm} keyed by m ≥ 0 (R_pos) and m < 0 (R_neg)
                key_base = (gaunt_l_val * (gaunt_l_val + 1)) ÷ 2 + Int32(1) # This is the key without the m offset.
                if gaunt_m_val >= 0
                    key_store = key_base + gaunt_m_val
                    CUDA.@atomic R_pos_real[transition_idx, q_idx, key_store] += real(R_val)
                    CUDA.@atomic R_pos_imag[transition_idx, q_idx, key_store] += imag(R_val)
                else
                    key_store = key_base - gaunt_m_val # +m is equivalent to -(-m).
                    CUDA.@atomic R_neg_real[transition_idx, q_idx, key_store] += real(R_val)
                    CUDA.@atomic R_neg_imag[transition_idx, q_idx, key_store] += imag(R_val)
                end
            end
        end

        idx += stride
    end
end


@inline function compute_spherical_harmonics(L_max::Int, R_ij_hat::CuArray{T, 3}, Ylm_norms::CuArray{T, 1}, ij_i::CuArray{Int32}, ij_j::CuArray{Int32}) where {T<:AbstractFloat}
    """
    Compute the spherical harmonics on the GPU for all (i, j) pairs up to L_max, and return them in a CuArray.
    These are defined in terms of the associated Legendre polynomials P_L^M(cos(θ)) as:

        Y_L^M(θ, ϕ) = sqrt((2L + 1)/(4π) * (L - M)!/(L + M)!) * P_L^M(cos(θ)) * exp(i M ϕ),

    with the associated Legendre polynomials satisfying:

        P_0^0(x) = 1, P_m^m(x) = -(2m - 1) sqrt(1 - x^2) P_{m-1}^{m-1}(x), P_{m+1}^m(x) = (2m + 1) x P_m^m(x),
        P_l^m(x) = ((2l - 1) x P_{l-1}^m(x) - (l + m - 1) P_{l-2}^m(x)) / (l - m),

    which will be our computation strategy. To save on memory requirements, we store them with a linear index such that
    L geq |M|. We will also exploit the symmetry relation later in the code:

        Y_L^{-M}(θ, ϕ) = (-1)^M conj(Y_L^M(θ, ϕ)),

    and only store M geq 0 values. The key for accessing SHM (L,M) is:

        key(L, M) = (L * (L + 1)) / 2 + M + 1.

    # Arguments:
    - L_max::Int: The maximum L value to compute.
    - R_ij_hat::CuArray{T, 3}: The (θ, ϕ) angles for the unit vectors Rhat_{ij}.
    - Ylm_norms::CuArray{T, 1}: The precomputed normalisation constants for the spherical harmonics.
    - ij_i::CuArray{Int32}: The i indices for the non-zero (i, j) bins.
    - ij_j::CuArray{Int32}: The j indices for the non-zero (i, j) bins.

    # Returns:
    - CuArray{Complex{T}, 2}: The computed spherical harmonics tensor with dimensions (n_LM, n_nonzero_ij_bins).
    """

    # Allocate the output array on the GPU.
    n_bins = size(ij_i, 1)
    Ylm_tensor = CUDA.zeros(Complex{T}, n_bins, ((L_max + 1) * (L_max + 2)) ÷ 2) # Layout (bin, key) for faster kernel writes.

    # Launch the kernel to compute the spherical harmonics.
    threads = 256
    blocks = cld(n_bins, threads)
    @cuda threads=threads blocks=blocks sphercial_harmonics_kernel!(Ylm_tensor, Ylm_norms, L_max, R_ij_hat, ij_i, ij_j)

    # Transpose to (key, bin) layout for coalesced (this means faster) access in the R kernel.
    Ylm_tensor_t = CUDA.zeros(eltype(Ylm_tensor), size(Ylm_tensor, 2), size(Ylm_tensor, 1))
    CUDA.permutedims!(Ylm_tensor_t, Ylm_tensor, (2, 1))

    return Ylm_tensor_t

end

function weighted_spherical_bessel_kernel!(weighted_bessel_tensor::CuDeviceArray{T, 3}, q_vals::CuDeviceVector{T}, L_max::Int, sigma_ij::CuDeviceMatrix{T}, R_ij_mod::CuDeviceMatrix{T}, ij_i::CuDeviceVector{Int32}, ij_j::CuDeviceVector{Int32}) where {T<:AbstractFloat}
    """
    The CUDA kernel to compute the weighted spherical Bessel functions.

    # Arguments:
    - weighted_bessel_tensor::CuArray{T, 3}: The output tensor to store the weighted spherical Bessel functions.
    - q_vals::CuArray{T, 1}: The q values stored on the GPU.
    - L_max::Int: The maximum L value to compute.
    - sigma_ij::CuArray{T, 2}: The σ_ij values stored on the GPU.
    - R_ij_mod::CuArray{T, 2}: The |R_ij| values stored on the GPU.
    - ij_i::CuArray{Int32}: The i indices for the non-zero (i, j) bins.
    - ij_j::CuArray{Int32}: The j indices for the non-zero (i, j) bins.
    """

    # Get the global thread index and stride.
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    n_bins = size(ij_i, 1)
    n_q = size(q_vals, 1)
    total = n_bins * n_q
    
    # Get the q and (i, j) values for this thread.
    @inbounds while idx <= total
        bin_idx = div(idx - 1, n_q) + 1
        q_idx = mod(idx - 1, n_q) + 1

        i_idx = ij_i[bin_idx]
        j_idx = ij_j[bin_idx]

        q_val = q_vals[q_idx]
        sigma_ij_val = sigma_ij[i_idx, j_idx]
        R_ij_mod_val = R_ij_mod[i_idx, j_idx]

        # Give each thread its own buffer to store the j_L values.
        x_f64 = Float64(q_val) * Float64(R_ij_mod_val)
        start_L = L_max + max(25, Int(ceil(abs(x_f64))))
        start_L = min(start_L, MAX_L_GLOBAL - 1) # Clamp to buffer size to avoid overflow.
        j_L_buffer = MVector{MAX_L_GLOBAL + 1, Float64}(undef)

        # Also precompute the Gaussian factor.
        gaussian_factor = exp(-0.5 * Float64(sigma_ij_val * sigma_ij_val * q_val * q_val))

        # Handle very small x separately to avoid the instability of the Miller recursion.
        if abs(x_f64) <= SMALL_X_THRESHOLD
            @inbounds j_L_buffer[1] = 1.0 # j_0(0) = 1.
            for L in 1:L_max
                @inbounds j_L_buffer[L + 1] = 0.0 # j_L(0) = 0 for L > 0.
            end
        else
            # Precompute expensive terms that will be reused many times.
            sin_x, cos_x = sincos(x_f64)
            inv_x = 1.0 / x_f64
            inv_xsq = inv_x * inv_x

            # Start well above L_max so that the downwards recursion is accurate.
            # Miller's algorithm requires starting at L >> L_max, so we use a buffer of 25.
            j_L_plus1 = 0.0
            j_L = 1.0

            # Now we recurse down to L = 0, using Miller's algorithm.
            @inbounds for L in start_L:-1:1
                # Store the relevant j_L values.
                if L <= L_max
                    j_L_buffer[L + 1] = j_L
                end
                j_L_minus1 = ((2 * L + 1) * inv_x) * j_L - j_L_plus1
                j_L_plus1 = j_L
                j_L = j_L_minus1
            end

            # Set the L = 0 value.
            j_L_buffer[1] = j_L

            # Now we normalise the results using j_0(x) = sin(x)/x, and j_1(x) = (sin(x) - x cos(x)) / x^2.
            j0_exact = sin_x * inv_x
            j1_exact = (sin_x - x_f64 * cos_x) * inv_xsq
            norm_numerator = j0_exact
            norm_denominator = j_L

            # By default, we normalise using j_0_exact/j_0.
            # However, if j_0 is very small, dividing by it can lead to numerical instability.
            # In this case, we switch to normalising using j_1_exact/j_1 instead.
            if abs(j1_exact) > abs(j0_exact) || abs(j0_exact) < eps(Float64)
                norm_numerator = j1_exact
                norm_denominator = j_L_plus1
            end

            # Finally, rescale everything and cast to type T for storage.
            norm = norm_numerator / norm_denominator
            @inbounds for L in 0:L_max
                j_L_buffer[L + 1] = j_L_buffer[L + 1] * norm
            end
        end

        # Now store the weighted spherical Bessel functions.
        @inbounds for L in 0:L_max
            weighted_bessel_tensor[L + 1, bin_idx, q_idx] = T(j_L_buffer[L + 1] * gaussian_factor)
        end
        idx += stride
    end
end


@inline function compute_weighted_bessel_functions(q_vals::CuArray{T}, L_max::Int, sigma_ij::CuArray{T}, R_ij_mod::CuArray{T}, ij_i::CuArray{Int32}, ij_j::CuArray{Int32}) where {T<:AbstractFloat}
    """
    Compute the spherical Bessel functions j_L(x) for L = 0:L_max using Miller's algorithm for
    downward recurrence:

        j_{L-1}(x) = ((2L + 1)/x) * j_L(x) - j_{L+1}(x),

    and store them on the GPU for all non-zero (i, j) bins and q values. We also weight them by the Gaussian factor:

        g(q, i, j) = exp(-0.5 * σ_ij^2 * q^2),

    to save recomputing this for later.

    # Arguments:
    - q_vals::CuDeviceVector{T}: The q values stored on the GPU.
    - L_max::Int: The maximum L value to compute the spherical Bessel functions for.
    - sigma_ij::CuDeviceMatrix{T}: The σ_ij values stored on the GPU.
    - R_ij_mod::CuDeviceMatrix{T}: The |R_ij| values stored on the GPU.
    - ij_i::CuDeviceVector{Int32}: The i indices for the non-zero (i, j) bins.
    - ij_j::CuDeviceVector{Int32}: The j indices for the non-zero (i, j) bins.

    # Returns:
    - weighted_bessel_tensor::CuArray{T, 3}: The weighted spherical Bessel functions with dimensions (L_max + 1, n_nonzero_ij_bins, n_q).
    """
    
    # Allocate the output array on the GPU.
    n_bins = size(ij_i, 1)
    n_q = size(q_vals, 1)
    weighted_bessel_tensor = CUDA.zeros(T, L_max + 1, n_bins, n_q) # Layout (L, bin, q) for coalesced access when iterating L.

    # Launch a kernel to compute the weighted spherical Bessel functions.
    threads = 256
    blocks = cld(n_bins * n_q, threads)
    @cuda threads=threads blocks=blocks weighted_spherical_bessel_kernel!(weighted_bessel_tensor, q_vals, L_max, sigma_ij, R_ij_mod, ij_i, ij_j)

    return weighted_bessel_tensor
end

function construct_R_tensor_gpu(
        W_tensor::SparseWTensor{T},
        sigma_ij::Array{T, 2},
        R_ij_mod::Array{T, 2},
        R_ij_hat::Array{T, 3},
        q_grid::Vector{T},
        l_max::Int,
        lambda_max::Int,
        n_max::Int,
        gaunt_array_path::String,
        transition_matrices::Vector{Matrix{T}},
        cartesian_term_to_orbital::Vector{Int};
        threshold::T = zero(T),
        ij_chunk_size::Int = 50000,
        w_chunk_size::Int = 256,
    ) where {T<:AbstractFloat}
    """
    Construct the R_{ℓm}(q) tensor defined by:

        R_{ℓm}(q) = 2 * √2 * (2π)^(5 / 2) * ∑_{ij} exp(-σ_{ij}^2 q^2/2) ∑_{L} i^L j_L(q R_{ij})
                  * ∑_{n} q^n ∑_{λ,μ} W_{ij,λμ}^{n} G_{λLℓ}^{μm} conj(Y_L^{m-μ}(Rhat_{ij})),

    where W_{ij,λμ}^{n} is the W tensor, G_{λLℓ}^{μm} are Gaunt coefficients, j_L are spherical Bessel functions,
    and Y_L^M are spherical harmonics evaluated at Rhat_{ij}. For efficiency, we split the R into the positive and
    negative m components, and collapse the (ℓ, m) indices into a single linear key defined by:

        key(ℓ, m) = (ℓ(ℓ + 1))/2 + m + 1.

    For m < 0, we find the correct entry with |m|.

    # Arguments:
    - W_tensor::SparseWTensor{T}: The sparse W tensor stored on the CPU.
    - sigma_ij::Array{T,2}: The σ_{ij} values for all Cartesian term pairs.
    - R_ij_mod::Array{T,2}: The |R_{ij}| distances for all Cartesian term pairs.
    - R_ij_hat::Array{T,3}: The (θ, ϕ) angles for the unit vectors Rhat_{ij}.
    - q_grid::Vector{T}: The 1D grid of |q| values at which to evaluate the R tensor, in keV.
    - l_max::Int: Maximum ℓ to include in the expansion.
    - lambda_max::Int: Maximum λ for the molecule.
    - n_max::Int: Maximum n for the molecule.
    - gaunt_array_path::String: Path to the precomputed Gaunt coefficients (HDF5 file).
    - transition_matrices::Vector{Matrix{T}}: The transition matrices.
    - cartesian_term_to_orbital::Vector{Int}: Mapping from Cartesian term index to orbital index.
    - threshold::T: A threshold value below which contributions satisfying (|TDM * W / max(TDM * W)| < threshold) will be discarded (default: 0.0).
    - w_chunk_size::Int: Number of W entries processed per thread to improve arithmetic intensity (default: 256).

    # Returns:
    - R_tensor_pos::CuArray{Complex{T}, 3}: The R tensor for m ≥ 0 stored on the GPU with dimensions (n_transitions, n_q, n_keys_pos).
    - R_tensor_neg::CuArray{Complex{T}, 3}: The R tensor for m < 0 stored on the GPU with dimensions (n_transitions, n_q, n_keys_pos).
    """

    # Load Gaunt coefficients.
    gaunt_array = load_gaunt_array(gaunt_array_path, T)
    gaunt_coeffs = gaunt_array.coefficients

    # Preallocate the R tensor.
    L_max = l_max + lambda_max
    n_q = length(q_grid)
    n_transitions = length(transition_matrices)

    if L_max > MAX_L_GLOBAL - 1
        error("L_max = $(L_max) exceeds MAX_L_GLOBAL - 1 = $(MAX_L_GLOBAL - 1). Increase MAX_L_GLOBAL or reduce l_max.")
    end

    if L_max > MAX_L_GLOBAL - 1
        error("L_max = $(L_max) exceeds MAX_L_GLOBAL - 1 = $(MAX_L_GLOBAL - 1). Increase MAX_L_GLOBAL or reduce l_max.")
    end

    # Convert the q_grid from keV to inverse Å without excessive allocations.
    q_grid_invA = Vector{T}(undef, n_q)
    unit_conversion = T(KEV_TO_INV_ANGSTROM)
    @inbounds for i in 1:n_q
        q_grid_invA[i] = q_grid[i] * unit_conversion
    end

    # Precompute the powers of q on the CPU and move to the GPU.
    q_powers_gpu = compute_q_powers(q_grid_invA, n_max, n_q)

    # Compute the i and -1 powers on CPU and transfer them to the GPU.
    i_powers_cpu = [fast_i_pow(L, T) for L in 0:L_max]
    i_powers_gpu = CuArray(i_powers_cpu)
    neg1_powers_cpu = [fast_neg1_pow(M, T) for M in -L_max:L_max]
    neg1_powers_gpu = CuArray(neg1_powers_cpu)

    q_grid_gpu = CuArray{T,1}(q_grid_invA)

    # Allocate the R tensor on the GPU in flattened (transition, q, lm_key) layout for m ≥ 0 and m < 0 separately.
    # We use the same key as for the spherical harmonics, key(l, m) = (l(l+1))/2 + m + 1, and use |m| for the m < 0 case.
    n_keys_pos = (l_max + 1) * (l_max + 2) ÷ 2
    R_tensor_pos_real = CUDA.zeros(T, n_transitions, n_q, n_keys_pos)
    R_tensor_pos_imag = CUDA.zeros(T, n_transitions, n_q, n_keys_pos)
    R_tensor_neg_real = CUDA.zeros(T, n_transitions, n_q, n_keys_pos)
    R_tensor_neg_imag = CUDA.zeros(T, n_transitions, n_q, n_keys_pos)

    # Move the (i,j) objects to the GPU.
    sigma_ij_gpu = CuArray(sigma_ij)
    R_ij_mod_gpu = CuArray(R_ij_mod)
    R_ij_hat_gpu = CuArray(R_ij_hat) # The last dimension is (R_θ, R_ϕ).

    # Now construct the GPU version of the W tensor.
    # First, allocate the index and offset arrays on the CPU.
    n_nonzero_W = length(W_tensor.W_values)
    n_ij_bins = length(W_tensor.ij_bins)
    W_bin_idx_cpu = Vector{Int32}(undef, n_nonzero_W)
    # Also store the i and j indices of the non-zero bins.
    ij_i_cpu = Int32[]
    ij_j_cpu = Int32[]
    # Keep track of the mapping from compact bin index -> original (i, j) bin index.
    compact_to_full_bin = Int32[]

    # Fill the index arrays for the (i, j) bins.
    current_position = 1
    compact_idx = Int32(0)
    @inbounds for bin_idx in 1:n_ij_bins
        bin = W_tensor.ij_bins[bin_idx]
        if isempty(bin) 
            continue
        end
        compact_idx += Int32(1)

        # Store the i and j indices for this bin.
        push!(ij_i_cpu, W_tensor.i[bin[1]])
        push!(ij_j_cpu, W_tensor.j[bin[1]])
        push!(compact_to_full_bin, Int32(bin_idx))

        # Now fill in the indices for this bin.
        for _ in bin
            W_bin_idx_cpu[current_position] = compact_idx
            current_position += 1
        end
    end

    # Precompute TDM prefactors and transition-specific thresholds for each (i, j) bin.
    n_compact_bins = length(ij_i_cpu)
    W_max = W_tensor.W_max
    max_TDM_W = fill(zero(T), n_transitions)
    tdm_prefactors_cpu = Array{T, 2}(undef, n_transitions, n_compact_bins)

    @inbounds for bin_idx in 1:n_compact_bins
        pair_i = ij_i_cpu[bin_idx]
        pair_j = ij_j_cpu[bin_idx]
        orbital_i = cartesian_term_to_orbital[pair_i]
        orbital_j = cartesian_term_to_orbital[pair_j]

        for transition_idx in 1:n_transitions
            TDM = transition_matrices[transition_idx]
            # If i == j, we only take the diagonal element. Otherwise, sum both contributions.
            prefactor = pair_i == pair_j ? TDM[orbital_i, orbital_j] : TDM[orbital_i, orbital_j] + TDM[orbital_j, orbital_i]
            tdm_prefactors_cpu[transition_idx, bin_idx] = prefactor
            max_TDM_W[transition_idx] = max(max_TDM_W[transition_idx], abs(prefactor) * W_max)
        end
    end

    threshold_values_cpu = similar(max_TDM_W)
    @inbounds for transition_idx in 1:n_transitions
        threshold_values_cpu[transition_idx] = threshold * max_TDM_W[transition_idx]
    end

    # Move the threshold values to GPU.
    threshold_values_gpu = CuArray{T}(threshold_values_cpu)

    # Next, move the Gaunt coefficients to the GPU.
    # First we define the index and offset array for the (λ, μ) bins.
    n_nonzero_gaunt = length(gaunt_coeffs)
    n_lambda_mu_bins = length(gaunt_array.lambda_mu_bins)
    lambda_mu_indices_cpu = Vector{Int32}(undef, n_nonzero_gaunt)
    lambda_mu_offsets_cpu = Vector{Int32}(undef, n_lambda_mu_bins + 1)

    # Fill the index and offset arrays for the (λ, μ) bins.
    current_position = 1
    @inbounds for bin_idx in 1:n_lambda_mu_bins
        # Set the start of the bin.
        lambda_mu_offsets_cpu[bin_idx] = current_position
        bin = gaunt_array.lambda_mu_bins[bin_idx]
        # Now fill in the indices for this bin.
        for entry_idx in bin
            lambda_mu_indices_cpu[current_position] = Int32(entry_idx)
            current_position += 1
        end
    end
    lambda_mu_offsets_cpu[n_lambda_mu_bins + 1] = current_position

    # Now move the remaining Gaunt data to the GPU.
    lambda_mu_indices_gpu = CuArray{Int32}(lambda_mu_indices_cpu)
    lambda_mu_offsets_gpu = CuArray{Int32}(lambda_mu_offsets_cpu)
    gaunt_mu_gpu = CuArray{Int32}(gaunt_array.mu)
    gaunt_L_gpu = CuArray{Int32}(gaunt_array.L)
    gaunt_l_gpu = CuArray{Int32}(gaunt_array.l)
    gaunt_m_gpu = CuArray{Int32}(gaunt_array.m)
    gaunt_coeffs_gpu = CuArray{T}(gaunt_coeffs)

    # Compute the spherical harmonics on the GPU. We only do this for the non-zero (i, j) bins.
    # First precompute the normalisation constants, and move them to the GPU.
    Ylm_norms = precompute_spherical_harmonic_normalisation(T, L_max)
    Ylm_norms_gpu = CuArray(Ylm_norms)

    # Warm up GPU kernels to avoid first-launch overhead in profiling.
    @cuda threads=1 blocks=1 sphercial_harmonics_kernel!(CUDA.zeros(Complex{T}, 1, ((L_max + 1) * (L_max + 2)) ÷ 2), Ylm_norms_gpu, 0, CUDA.zeros(T, 1, 1, 2), CuArray{Int32}([1]), CuArray{Int32}([1]))
    @cuda threads=1 blocks=1 weighted_spherical_bessel_kernel!(CUDA.zeros(T, 1, 1, 1), CuArray{T}([zero(T)]), 0, CUDA.zeros(T, 1, 1), CUDA.zeros(T, 1, 1), CuArray{Int32}([1]), CuArray{Int32}([1]))
    @cuda threads=1 blocks=1 R_tensor_kernel!(
        CUDA.zeros(T, 1, 1, 1),
        CUDA.zeros(T, 1, 1, 1),
        CUDA.zeros(T, 1, 1, 1),
        CUDA.zeros(T, 1, 1, 1),
        CUDA.zeros(Complex{T}, ((L_max + 1) * (L_max + 2)) ÷ 2, 1),
        CUDA.zeros(T, L_max + 1, 1, 1),
        CuArray{Complex{T}}([zero(Complex{T})]),
        CuArray{Int32}([1]),
        CuArray{Int32}([1]),
        CuArray{Int32}([0]),
        CuArray{Int32}([1, 2]),
        CuArray{T}([zero(T)]),
        CuArray{Int32}([1]),
        CuArray{Int32}([1, 1]),
        CuArray{Int32}([0]),
        CuArray{Int32}([0]),
        CuArray{Int32}([0]),
        CuArray{Int32}([0]),
        CUDA.zeros(T, 1, 1),
        CuArray{Complex{T}}([one(Complex{T})]),
        CuArray{T}([one(T)]),
        CuArray{T}(reshape([one(T)], 1, 1)),
        CuArray{T}([zero(T)]),
        Int32(1),
        Int32(1),
        Int32(l_max),
        Int32(lambda_max),
        Int32(L_max),
        Int64(1),
    )

    # Process ij bins in chunks to reduce the memory usage of intermediate tensors.
    n_chunks = cld(n_compact_bins, ij_chunk_size)
    println("Processing $(n_compact_bins) ij bins in $(n_chunks) chunk(s) of size $(ij_chunk_size)")

    for chunk_idx in 1:n_chunks
        # Determine the range of bins for this chunk.
        bin_start = (chunk_idx - 1) * ij_chunk_size + 1
        bin_end = min(chunk_idx * ij_chunk_size, n_compact_bins)

        # Extract the ij indices for this chunk, and move them to the GPU.
        ij_i_chunk = ij_i_cpu[bin_start:bin_end]
        ij_j_chunk = ij_j_cpu[bin_start:bin_end]
        ij_i_chunk_gpu = CuArray{Int32}(ij_i_chunk)
        ij_j_chunk_gpu = CuArray{Int32}(ij_j_chunk)

        # Filter W entries that belong to this chunk of bins.
        W_chunk_lambda = Int32[]
        W_chunk_mu = Int32[]
        W_chunk_n = Int32[]
        W_chunk_values = Complex{T}[]
        W_bin_offsets_chunk = Int32[1]

        @inbounds for bin_global in bin_start:bin_end
            original_bin_idx = compact_to_full_bin[bin_global]
            bin = W_tensor.ij_bins[original_bin_idx]
            for w_idx in bin
                push!(W_chunk_lambda, W_tensor.lambda[w_idx])
                push!(W_chunk_mu, W_tensor.mu[w_idx])
                push!(W_chunk_n, W_tensor.n[w_idx])
                push!(W_chunk_values, W_tensor.W_values[w_idx])
            end
            push!(W_bin_offsets_chunk, length(W_chunk_lambda) + 1)
        end

        n_W_chunk = length(W_chunk_lambda)
        n_bins_chunk = length(W_bin_offsets_chunk) - 1

        # Skip empty chunks.
        n_W_chunk == 0 && continue

        # Move the chunk-specific W data to GPU.
        W_lambda_chunk_gpu = CuArray{Int32}(W_chunk_lambda)
        W_mu_chunk_gpu = CuArray{Int32}(W_chunk_mu)
        W_n_chunk_gpu = CuArray{Int32}(W_chunk_n)
        W_values_chunk_gpu = CuArray{Complex{T}}(W_chunk_values)
        W_bin_offsets_chunk_gpu = CuArray{Int32}(W_bin_offsets_chunk)

        # Move the chunk-specific W data to GPU.
        # Extract the TDM prefactors for this chunk.
        tdm_prefactors_chunk = tdm_prefactors_cpu[:, bin_start:bin_end]
        tdm_prefactors_chunk_gpu = CuArray{T}(tdm_prefactors_chunk)

        # Compute the spherical harmonics tensor for this chunk.
        Ylm_tensor_chunk_gpu = compute_spherical_harmonics(L_max, R_ij_hat_gpu, Ylm_norms_gpu, ij_i_chunk_gpu, ij_j_chunk_gpu)

        # Compute the weighted spherical Bessel functions for this chunk.
        weighted_bessel_chunk_gpu = compute_weighted_bessel_functions(q_grid_gpu, L_max, sigma_ij_gpu, R_ij_mod_gpu, ij_i_chunk_gpu, ij_j_chunk_gpu)

        # Compute R using the W-major kernel for this chunk.
        threads = 256
        total_wq_chunk = Int64(n_bins_chunk) * Int64(n_q) * Int64(n_transitions)
        blocks = cld(total_wq_chunk, threads)
        @cuda threads=threads blocks=blocks R_tensor_kernel!(
            R_tensor_pos_real, R_tensor_pos_imag, R_tensor_neg_real, R_tensor_neg_imag,
            Ylm_tensor_chunk_gpu, weighted_bessel_chunk_gpu,
            W_values_chunk_gpu, W_lambda_chunk_gpu, W_mu_chunk_gpu, W_n_chunk_gpu, W_bin_offsets_chunk_gpu,
            gaunt_coeffs_gpu, lambda_mu_indices_gpu, lambda_mu_offsets_gpu,
            gaunt_mu_gpu, gaunt_L_gpu, gaunt_l_gpu, gaunt_m_gpu,
            q_powers_gpu, i_powers_gpu, neg1_powers_gpu,
            tdm_prefactors_chunk_gpu, threshold_values_gpu,
            Int32(n_q), Int32(n_bins_chunk), Int32(l_max), Int32(lambda_max), Int32(L_max), total_wq_chunk
        )

        # Free the GPU memory for this chunk before moving to the next.
        CUDA.unsafe_free!(Ylm_tensor_chunk_gpu)
        CUDA.unsafe_free!(weighted_bessel_chunk_gpu)
        CUDA.unsafe_free!(W_lambda_chunk_gpu)
        CUDA.unsafe_free!(W_mu_chunk_gpu)
        CUDA.unsafe_free!(W_n_chunk_gpu)
        CUDA.unsafe_free!(W_values_chunk_gpu)
        CUDA.unsafe_free!(W_bin_offsets_chunk_gpu)
        CUDA.unsafe_free!(tdm_prefactors_chunk_gpu)
        CUDA.unsafe_free!(ij_i_chunk_gpu)
        CUDA.unsafe_free!(ij_j_chunk_gpu)
    end

    # Free temporary GPU buffers that are no longer needed before returning the R tensors.
    CUDA.unsafe_free!(q_powers_gpu)
    CUDA.unsafe_free!(i_powers_gpu)
    CUDA.unsafe_free!(neg1_powers_gpu)
    CUDA.unsafe_free!(sigma_ij_gpu)
    CUDA.unsafe_free!(R_ij_mod_gpu)
    CUDA.unsafe_free!(R_ij_hat_gpu)
    CUDA.unsafe_free!(lambda_mu_indices_gpu)
    CUDA.unsafe_free!(lambda_mu_offsets_gpu)
    CUDA.unsafe_free!(gaunt_mu_gpu)
    CUDA.unsafe_free!(gaunt_L_gpu)
    CUDA.unsafe_free!(gaunt_l_gpu)
    CUDA.unsafe_free!(gaunt_m_gpu)
    CUDA.unsafe_free!(gaunt_coeffs_gpu)
    CUDA.unsafe_free!(Ylm_norms_gpu)
    CUDA.unsafe_free!(q_grid_gpu)
    CUDA.unsafe_free!(threshold_values_gpu)
    CUDA.reclaim()

    # Apply the overall prefactor before combining the real and imaginary parts.
    prefactor_T = T(prefactor)
    R_tensor_pos_real .*= prefactor_T
    R_tensor_pos_imag .*= prefactor_T
    R_tensor_neg_real .*= prefactor_T
    R_tensor_neg_imag .*= prefactor_T

    # Combine the real and imaginary parts.
    R_tensor_pos = Complex.(R_tensor_pos_real, R_tensor_pos_imag)
    R_tensor_neg = Complex.(R_tensor_neg_real, R_tensor_neg_imag)

    return R_tensor_pos, R_tensor_neg
end
end
