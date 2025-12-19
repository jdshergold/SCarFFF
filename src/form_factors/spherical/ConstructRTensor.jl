# This module contains functions to construct the R tensor from the W tensor.

module ConstructRTensor

using HDF5
using SphericalHarmonics
using Base.Threads

include("../../utils/BinEncoding.jl")

using .BinEncoding: decode_bins
using ...SparseTensors
using ...FastPowers: fast_i_pow
using ...SparseTensors: SparseWTensor, SparseGauntArray, lambda_mu_key

export construct_R_tensor

@inline function load_gaunt_array(path::String, ::Type{T}) where {T<:AbstractFloat}
    """
    Load a SparseGauntArray from an HDF5 file.

    This function reads the Gaunt coefficients and the associated indices from a HDF5 file
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
const prefactor = 2.0 * sqrt(2) * (2π)^(5 / 2)

@inline function fill_spherical_bessel_column!(
        j_L_matrix::Array{T, 2},
        j_vals_buffer::Vector{Float64},
        x_column::Int,
        x::T,
        L_max::Int,
    ) where {T<:AbstractFloat}
    """
    Compute the spherical Bessel functions j_L(x) for L = 0:L_max using Miller's algorithm for
    downward recurrence:

        j_{L-1}(x) = ((2L + 1)/x) * j_L(x) - j_{L+1}(x),

    and fill the specified column of j_L_matrix with the results. All calculations
    are performed in Float64 for stability, then cast back to T when stored.

    # Arguments:
    - j_L_matrix::Array{T, 2}: Preallocated array to store j_L(x) values.
    - j_vals_buffer::Vector{Float64}: Preallocated buffer for storing j_L values. This avoid allocations.
    - x_column::Int: The column index in j_L_matrix to fill.
    - x::T: The value at which to evaluate the spherical Bessel functions.
    - L_max::Int: The maximum order L of the spherical Bessel functions.
    """
    x_f64 = Float64(x)

    # Handle very small x separately to avoid the instability of the Miller recursion.
    if abs(x_f64) <= SMALL_X_THRESHOLD
        @inbounds j_vals_buffer[1] = 1.0 # j_0(0) = 1.
        @inbounds for row in 2:(L_max + 1)
            j_vals_buffer[row] = 0.0 # j_L(0) = 0 for L > 0.
        end
        @inbounds for row in 1:(L_max + 1)
            j_L_matrix[row, x_column] = T(j_vals_buffer[row])
        end
        return
    end

    # Precompute expensive terms that will be reused many times.
    sin_x, cos_x = sincos(x_f64)
    inv_x = 1.0 / x_f64
    inv_xsq = inv_x * inv_x

    # Start well above L_max so that the downwards recursion is accurate.
    # Miller's algorithm requires starting at L >> L_max, so we use a buffer of
    # max(25, ceil(x)) for robust performance.
    start_L = L_max + max(25, Int(ceil(x_f64)))
    j_L_plus1 = 0.0
    j_L = 1.0

    # Now we recurse down to L = 0, using Miller's algorithm.
    @inbounds for L in start_L:-1:1
        # Store the relevant j_L values.
        if L <= L_max
            j_vals_buffer[L + 1] = j_L
        end
        j_L_minus1 = ((2 * L + 1) * inv_x) * j_L - j_L_plus1
        j_L_plus1 = j_L
        j_L = j_L_minus1
    end

    # Set the L = 0 value.
    j_vals_buffer[1] = j_L

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
    return @inbounds for row in 1:(L_max + 1)
        j_L_matrix[row, x_column] = T(j_vals_buffer[row] * norm)
    end
end

@inline function fill_q_powers!(
        q_powers::Array{T, 2},
        q_grid::Vector{T},
        n_max::Int,
        n_q::Int,
    ) where {T<:AbstractFloat}
    """
    Helper function to compute powers of q from 0 to n_max.
    Modifies the q_powers array in-place to avoid allocations.

    # Arguments:
    - q_powers::Array{T,2}: Preallocated array to store q^n values.
    - q_grid::Vector{T}: The grid of q values.
    - n_max::Int: The maximum power of q to compute.
    - n_q::Int: The number of q values in the q_grid.

    # Returns:
    - Nothing, q_powers is modified in-place. Element q_powers[n+1, k] contains q_grid[k]^n.
    """

    typed_one = one(T)

    return @inbounds for q_idx in 1:n_q
        q = q_grid[q_idx]
        q_powers[1, q_idx] = typed_one
        for n in 1:n_max
            q_powers[n + 1, q_idx] = q_powers[n, q_idx] * q
        end
    end
end

@inline function fill_i_powers!(i_powers::Vector{Complex{T}}, L_max::Int) where {T<:AbstractFloat}
    """
    Helper function to compute powers of i from 0 to L_max.
    Modifies the i_powers array in-place to avoid allocations.

    # Arguments:
    - i_powers::Vector{Complex{T}}: Preallocated vector to store i^L values.
    - L_max::Int: The maximum power of i to compute.

    # Returns:
    - Nothing, i_powers is modified in-place. Element i_powers[L+1] contains i^L.
    """
    return @inbounds for L in 0:L_max
        i_powers[L + 1] = fast_i_pow(L, T)
    end
end

function construct_R_tensor(
        W_tensor::SparseWTensor{T},
        sigma_ij::Array{T, 2},
        R_ij_mod::Array{T, 2},
        R_ij_hat::Array{T, 3},
        q_grid::Vector{T},
        l_max::Int,
        gaunt_array_path::String,
        transition_matrices::Vector{Matrix{T}},
        cartesian_term_to_orbital::Vector{Int};
        threshold::T = zero(T),
    )::Array{Complex{T}, 3} where {T<:AbstractFloat}
    """
    Construct the R_{ℓm}(q) tensor defined by:

        R_{ℓm}(q) = 2 √2 (2π)^(5/2) ∑_{ij pairs} (TDM_ij + TDM_ji) * exp(-σ_{ij}^2 q^2/2) ∑_{L} i^L j_L(q R_{ij})
                  * ∑_{n} q^n ∑_{λ,μ} W_{ij,λμ}^{n} G_{λLℓ}^{μm} conj(Y_L^{m-μ}(Rhat_{ij})),

    where W_{ij,λμ}^{n} is the W tensor, TDM is the transition matrix, G_{λLℓ}^{μm} are Gaunt coefficients,
    j_L are spherical Bessel functions, and Y_L^M are spherical harmonics evaluated at Rhat_{ij}.

    This function batches over all transitions, computing R for each transition simultaneously with optional thresholding.

    # Arguments:
    - W_tensor::SparseWTensor{T}: The W tensor in sparse COO format.
    - sigma_ij::Array{T,2}: The σ_{ij} values for all Cartesian term pairs.
    - R_ij_mod::Array{T,2}: The |R_{ij}| distances for all Cartesian term pairs.
    - R_ij_hat::Array{T,3}: The (θ, ϕ) angles for the unit vectors Rhat_{ij}.
    - q_grid::Vector{T}: The 1D grid of |q| values at which to evaluate the R tensor, in keV.
    - l_max::Int: Maximum ℓ to include in the expansion.
    - gaunt_array_path::String: Path to the precomputed Gaunt coefficients (HDF5 file).
    - transition_matrices::Vector{Matrix{T}}: Vector of transition matrices, one per transition.
    - cartesian_term_to_orbital::Vector{Int}: Mapping from Cartesian term index to orbital index.
    - threshold::T: Threshold for skipping (i,j) pairs. If |TDM_ij + TDM_ji| * W_max < threshold * global_max, skip this pair (default: 0.0).

    # Returns:
    - R_tensor::Array{Complex{T},3}: The computed R tensor, with dimensions (n_transitions, q, lm_key), keyed by lm = ℓ^2 + (ℓ + m) + 1.
    """

    # Load Gaunt coefficients.
    gaunt_array = load_gaunt_array(gaunt_array_path, T)
    gaunt_coeffs = gaunt_array.coefficients

    # Get the relevant dimensions.
    lambda_max = maximum(W_tensor.lambda)
    L_max = l_max + lambda_max
    n_q = length(q_grid)
    n_transitions = length(transition_matrices)

    # Convert the q_grid from keV to inverse Å without excessive allocations.
    q_grid_invA = Vector{T}(undef, n_q)
    unit_conversion = T(KEV_TO_INV_ANGSTROM)
    @inbounds for i in 1:n_q
        q_grid_invA[i] = q_grid[i] * unit_conversion
    end

    # Preallocate the R tensor with a batch dimension for transitions.
    # To save memory, we use the triangular key for (ℓ, m).
    n_keys = (l_max + 1)^2
    R_tensor = zeros(Complex{T}, n_transitions, n_q, n_keys)

    # Get the number of (i, j) bins and number of threads.
    num_ij_bins = length(W_tensor.ij_bins)
    n_threads = nthreads()

    # Allocate arrays for precomputed quantities.
    n_max = W_tensor.n_max
    q_powers = Array{T, 2}(undef, n_max + 1, n_q)
    i_powers = Vector{Complex{T}}(undef, L_max + 1)

    # Precompute the powers of q.
    fill_q_powers!(q_powers, q_grid_invA, n_max, n_q)

    # Precompute the powers of i.
    fill_i_powers!(i_powers, L_max)

    # Prellocate buffers to each thread to avoid races.
    gaussian_pool = [Vector{T}(undef, n_q) for _ in 1:n_threads]
    jL_pool = [Array{T, 2}(undef, L_max + 1, n_q) for _ in 1:n_threads]
    jL_miller_pool = [Vector{Float64}(undef, L_max + 1) for _ in 1:n_threads] # This is a buffer for Miller's algorithm, to save repeated allocations when recursing downwards.
    Y_cache_pool = [SphericalHarmonics.cache(L_max, SphericalHarmonics.FullRange) for _ in 1:n_threads]
    R_local_pool = [zeros(Complex{T}, n_q, n_keys) for _ in 1:n_threads]

    typed_half = T(0.5)
    W_max = W_tensor.W_max

    # Loop over transitions.
    for transition_idx in 1:n_transitions
        # Extract TDM for this transition as a view to avoid allocations.
        TDM = @view transition_matrices[transition_idx][:, :]

        # Compute the maximum |TDM_ij + TDM_ji| * W_max for this transition for thresholding.
        max_TDM_W = zero(T)
        if threshold > zero(T)
            @inbounds for bin_idx in 1:num_ij_bins
                ij_bin = W_tensor.ij_bins[bin_idx]
                isempty(ij_bin) && continue

                first_idx = ij_bin[1]
                pair_i = W_tensor.i[first_idx]
                pair_j = W_tensor.j[first_idx]

                orbital_i = cartesian_term_to_orbital[pair_i]
                orbital_j = cartesian_term_to_orbital[pair_j]

                # Compute |TDM_ij + TDM_ji| * W_max, handling diagonal separately
                if pair_i == pair_j
                    TDM_contribution = abs(TDM[orbital_i, orbital_j]) * W_max
                else
                    TDM_contribution = abs(TDM[orbital_i, orbital_j] + TDM[orbital_j, orbital_i]) * W_max
                end

                # Update the maximum.
                max_TDM_W = max(max_TDM_W, TDM_contribution)
            end
        end

        # Compute the effective W threshold for this transition.
        threshold_value = threshold * max_TDM_W

        # Zero out the thread-local buffers for this transition.
        @inbounds for thread_id in 1:n_threads
            fill!(R_local_pool[thread_id], zero(Complex{T}))
        end

        # Accumulate one (i, j) slice at a time, threading over ij bins.
        @threads for bin_idx in 1:num_ij_bins
            ij_bin = W_tensor.ij_bins[bin_idx]

            # Skip empty (i, j) bins.
            isempty(ij_bin) && continue

            # Get thread-local buffers.
            thread_id = threadid()
            gaussian_local = gaussian_pool[thread_id]
            jL_local = jL_pool[thread_id]
            jL_miller_buffer = jL_miller_pool[thread_id]
            Ylm_cache = Y_cache_pool[thread_id]
            R_local = R_local_pool[thread_id]

            # All entries in this bin share the same (i, j) pair.
            first_idx = ij_bin[1]
            pair_i = W_tensor.i[first_idx]
            pair_j = W_tensor.j[first_idx]

            # Extract the orbital indices and compute the TDM prefactor for this pair.
            orbital_i = cartesian_term_to_orbital[pair_i]
            orbital_j = cartesian_term_to_orbital[pair_j]

            if pair_i == pair_j
                # For the diagonal entries there is only one contribution.
                TDM_prefactor = TDM[orbital_i, orbital_j]
            else
                TDM_prefactor = TDM[orbital_i, orbital_j] + TDM[orbital_j, orbital_i]
            end

            # Extract the geometry for this pair.
            sigma_ij_val = sigma_ij[pair_i, pair_j]
            R_mod = R_ij_mod[pair_i, pair_j]
            theta_ij = R_ij_hat[pair_i, pair_j, 1]
            phi_ij = R_ij_hat[pair_i, pair_j, 2]

            # Precompute the spherical harmonics for this pair.
            computePlmcostheta!(Ylm_cache, theta_ij, L_max)
            computeYlm!(Ylm_cache, theta_ij, phi_ij, L_max)
            Yvals = SphericalHarmonics.getY(Ylm_cache)

            # Compute Gaussian factor exp(-σ_{ij}^2 q^2/2), along with the spherical Bessel functions j_L(q R_{ij}).
            sigma_ij_sq = sigma_ij_val * sigma_ij_val
            @inbounds for q_idx in 1:n_q
                gaussian_local[q_idx] = exp(-typed_half * sigma_ij_sq * q_powers[3, q_idx])
                fill_spherical_bessel_column!(jL_local, jL_miller_buffer, q_idx, q_grid_invA[q_idx] * R_mod, L_max)
            end

            # Now loop over W entries in this (i, j) bin.
            @inbounds for W_idx in ij_bin
                # Extract the remaining indices, so that we can join with the corresponding Gaunt bin.
                lambda = W_tensor.lambda[W_idx]
                mu = W_tensor.mu[W_idx]
                n = W_tensor.n[W_idx]

                # Also extract the W tensor value.
                W_val = W_tensor.W_values[W_idx]

                # Apply thresholding.
                abs(TDM_prefactor) * abs(W_val) < threshold_value && continue

                n_idx = n + 1 # For future indexing.

                # Find the Gaunt bin for this (λ, μ).
                bin_key = lambda_mu_key[lambda + 1, mu + lambda + 1]
                gaunt_bin = gaunt_array.lambda_mu_bins[bin_key]

                # Skip if there are no Gaunt coefficients for this (λ, μ).
                isempty(gaunt_bin) && continue

                # Loop over the matching Gaunt coefficients.
                for gaunt_idx in gaunt_bin
                    L = gaunt_array.L[gaunt_idx]
                    l = gaunt_array.l[gaunt_idx]
                    m = gaunt_array.m[gaunt_idx]

                    gaunt_val = gaunt_coeffs[gaunt_idx]

                    # Compute the corresponding spherical harmonic index. M = m - μ.
                    M = m - mu

                    # Fetch the correct SHM and take its complex conjugate.
                    Y_val = Complex{T}(Yvals[(L, M)])
                    conj_Y = conj(Y_val)

                    # Compute the angular part of the contribution, including TDM prefactor.
                    angular_term = TDM_prefactor * W_val * gaunt_val * conj_Y * i_powers[L + 1]

                    # Precompute the key.
                    lm_key = l * l + (l + m) + 1

                    # Now accumulate into R_local.
                    @inbounds for k in 1:n_q
                        R_local[k, lm_key] += angular_term * gaussian_local[k] * q_powers[n_idx, k] * jL_local[L + 1, k]
                    end
                end
            end
        end

        # Accumulate the thread-local results into the global R tensor for this transition.
        @inbounds for R_per_thread in R_local_pool
            R_tensor[transition_idx, :, :] .+= R_per_thread
        end
    end

    # Apply the prefactor.
    R_tensor .*= Complex{T}(prefactor)

    return R_tensor
end

end
