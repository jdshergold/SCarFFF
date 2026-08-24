# This module contains functions to construct the R tensor from the W tensor.

module ConstructRTensor

using HDF5
using SphericalHarmonics
using Base.Threads
using LinearAlgebra: mul!

include("../../utils/BinEncoding.jl")

using .BinEncoding: decode_bins
using ...SparseTensors
using ...FastPowers: fast_i_pow, fast_neg1_pow
using ...SparseTensors: SparseWTensor, SparseGauntArray, lambda_mu_key
using ...ThreadChunks: chunk_count, chunk_range

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
# The first 2 is from the plane wave expansion. The second factor of 2
# accounts for the two spin channels, since the transition matrices carry the
# per-spin-channel X_α and Y_α straight from PySCF.
const prefactor = 2.0 * 2.0 * (2π)^(5 / 2)
# Maximum memory used by the real and imaginary blocks of S_{ℓm,ij}(q) together. The full set of
# Cartesian-term overlap coefficients is generally too large to store, so it is computed in blocks.
const PAIR_RESPONSE_BYTES = 64 * 1024 * 1024

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

function build_pair_density_weights(
        W_tensor::SparseWTensor{T},
        density_matrices::Vector{Matrix{T}},
        cartesian_term_to_orbital::Vector{Int},
    ) where {T<:AbstractFloat}
    """
    Convert the requested AO density-like matrices into weights for the non-empty Cartesian-term
    pairs stored in the W tensor. The density matrix weights, built from T_ij, appearing in:

        R_{lm}^{(a)}(q) = 2 ∑_{i,j} T_ij^(a) S_{ℓm,ij}(q) = 2 [∑_i T_ii^(a) S_{ℓm,ii}(q) + ∑_{j>i} (T_ij^(a) + T_ji^(a)) S_{ℓm,ij}(q)].

    is then returned, as appropriate. This is either T_ii or T_ij + T_ji. The factor of 2 for the
    two spin channels is applied later, when the R tensor is assembled.

    # Arguments:
    - W_tensor::SparseWTensor{T}: The sparse W tensor, binned by Cartesian-term pair (i, j).
    - density_matrices::Vector{Matrix{T}}: The AO density-like matrices used to form the requested
      R_{ℓm} coefficients.
    - cartesian_term_to_orbital::Vector{Int}: Mapping from Cartesian-term index to AO index.

    # Returns:
    - full_bins::Vector{Int}: Indices of the non-empty (i, j) bins in W_tensor.ij_bins.
    - pair_i::Vector{Int}: First Cartesian-term index for each non-empty pair.
    - pair_j::Vector{Int}: Second Cartesian-term index for each non-empty pair.
    - pair_density_weights::Matrix{T}: Density weight for every requested matrix and Cartesian pair,
      with dimensions (n_densities, n_pairs).
    """

    # W stores only the upper triangle of the Cartesian-term pair matrix. Convert each AO density
    # matrix into weights for those pairs without assuming that the AO matrix itself is symmetric.
    full_bins = Int[]
    pair_i = Int[]
    pair_j = Int[]

    # First make a list of the non-empty W bins, since thresholding W may have left empty (i, j) bins.
    for bin_idx in eachindex(W_tensor.ij_bins)
        ij_bin = W_tensor.ij_bins[bin_idx]
        isempty(ij_bin) && continue

        # All entries in this bin share the same Cartesian-term pair.
        first_idx = ij_bin[1]
        push!(full_bins, bin_idx)
        push!(pair_i, W_tensor.i[first_idx])
        push!(pair_j, W_tensor.j[first_idx])
    end

    # Precompute the density weights once.
    pair_density_weights = Matrix{T}(undef, length(density_matrices), length(full_bins))
    @inbounds for pair_idx in eachindex(full_bins)
        i = pair_i[pair_idx]
        j = pair_j[pair_idx]
        orbital_i = cartesian_term_to_orbital[i]
        orbital_j = cartesian_term_to_orbital[j]

        for density_idx in eachindex(density_matrices)
            density = density_matrices[density_idx]

            # If i = j, return the diagonal element, otherwise the sum across the diagonal.
            pair_density_weights[density_idx, pair_idx] = i == j ?
                density[orbital_i, orbital_j] :
                density[orbital_i, orbital_j] + density[orbital_j, orbital_i]
        end
    end
    return full_bins, pair_i, pair_j, pair_density_weights
end

function construct_R_tensor(
        W_tensor::SparseWTensor{T},
        sigma_ij::Array{T, 2},
        R_ij_mod::Array{T, 2},
        R_ij_hat::Array{T, 3},
        q_grid::Vector{T},
        l_max::Int,
        gaunt_array_path::String,
        density_matrices::Vector{Matrix{T}},
        cartesian_term_to_orbital::Vector{Int},
    )::Array{Complex{T}, 3} where {T<:AbstractFloat}
    """
    Construct the R_{ℓm}(q) tensors for all of the requested density-like matrices:

        R_{ℓm}^{(a)}(q) = 2 [∑_i T_ii^(a) S_{ℓm,ii}(q) + ∑_{j>i} (T_ij^(a) + T_ji^(a)) S_{ℓm,ij}(q)].

    Here, the factor of 2 accounts for the two spin channels, and the Cartesian-term overlap
    coefficient is

        S_{ℓm,ij}(q) = 2 (2π)^(5/2) exp(-σ_{ij}²q²/2) ∑_{L,n,λ,μ} i^L j_L(qR_{ij}) q^n
                         W_{ij,λμ}^n G_{λLℓ}^{μm} conj(Y_L^{m-μ}(Rhat_{ij})),

    where W_{ij,λμ}^n is the W tensor, G_{λLℓ}^{μm} are Gaunt coefficients, j_L are spherical Bessel
    functions, and Y_L^M are spherical harmonics evaluated at Rhat_{ij}. Each S_{ℓm,ij}(q) is calculated
    once, in blocks, before being combined with every density matrix. Only m ≥ 0 is calculated
    directly, since

        R_{ℓ,-m}^{(a)}(q) = (-1)^(m-ℓ) R_{ℓm}^{(a)*}(q),

    lets us skip half the work.

    # Arguments:
    - W_tensor::SparseWTensor{T}: The sparse W tensor, already filtered by the requested W threshold.
    - sigma_ij::Array{T,2}: The σ_{ij} values for all Cartesian-term pairs.
    - R_ij_mod::Array{T,2}: The |R_{ij}| distances for all Cartesian-term pairs.
    - R_ij_hat::Array{T,3}: The (θ, ϕ) angles of the unit vectors Rhat_{ij}.
    - q_grid::Vector{T}: The one-dimensional grid of |q| values in keV.
    - l_max::Int: Maximum ℓ to include in the expansion.
    - gaunt_array_path::String: Path to the precomputed sparse Gaunt coefficients.
    - density_matrices::Vector{Matrix{T}}: The AO density-like matrices used to construct R. These
      may be transition matrices or diagonal-correction matrices and are not assumed symmetric.
    - cartesian_term_to_orbital::Vector{Int}: Mapping from Cartesian-term index to AO index.

    # Returns:
    - R_tensor::Array{Complex{T},3}: The prefactored R tensor with dimensions
      (n_densities, n_q, (ℓ_max + 1)^2), keyed by key(ℓ,m) = ℓ^2 + (ℓ + m) + 1.
    """

    # Load the Gaunt coefficients.
    gaunt_array = load_gaunt_array(gaunt_array_path, T)
    gaunt_coeffs = gaunt_array.coefficients

    # Get the relevant angular and output dimensions. Only the triangular m ≥ 0 range is computed
    # explicitly, which halves both the S coefficient work and its temporary storage.
    lambda_max = maximum(W_tensor.lambda)
    L_max = l_max + lambda_max
    n_q = length(q_grid)
    n_densities = length(density_matrices)
    n_keys_pos = (l_max + 1) * (l_max + 2) ÷ 2 # Number of keys with m ≥ 0..
    n_outputs = n_q * n_keys_pos

    # Convert the q grid from keV to inverse Angstroms without excessive allocations.
    q_grid_invA = Vector{T}(undef, n_q)
    unit_conversion = T(KEV_TO_INV_ANGSTROM)
    @inbounds @simd for q_idx in eachindex(q_grid)
        q_grid_invA[q_idx] = q_grid[q_idx] * unit_conversion
    end

    # Allocate and precompute the powers of q and i that are reused for every Cartesian pair.
    n_max = W_tensor.n_max
    q_powers = Matrix{T}(undef, n_max + 1, n_q)
    i_powers = Vector{Complex{T}}(undef, L_max + 1)

    # Precompute the powers of q.
    fill_q_powers!(q_powers, q_grid_invA, n_max, n_q)

    # Precompute the powers of i.
    fill_i_powers!(i_powers, L_max)

    # List the non-empty W bins and prepare the density weight (T_ij factors) for each Cartesian pair before the
    # expensive S calculation.
    full_bins, pair_i, pair_j, pair_density_weights = build_pair_density_weights(
        W_tensor, density_matrices, cartesian_term_to_orbital)
    n_pairs = length(full_bins)

    # Keep the real and imaginary blocks of S within a fixed memory budget.
    bytes_per_pair = 2 * n_outputs * sizeof(T)
    pair_block = min(n_pairs, max(1, PAIR_RESPONSE_BYTES ÷ bytes_per_pair))
    response_real = Matrix{T}(undef, n_outputs, pair_block)
    response_imag = Matrix{T}(undef, n_outputs, pair_block)

    # Preallocate one set of buffers per task rather than per thread, so every buffer has exactly
    # one writer even if a task moves between Julia threads. The buffers are reused for every block.
    max_tasks = chunk_count(pair_block, nthreads())
    gaussian_pool = [Vector{T}(undef, n_q) for _ in 1:max_tasks]
    jL_pool = [Matrix{T}(undef, L_max + 1, n_q) for _ in 1:max_tasks]
    # This buffer is reused by Miller's algorithm rather than allocated for every pair and q value.
    jL_miller_pool = [Vector{Float64}(undef, L_max + 1) for _ in 1:max_tasks]
    Y_cache_pool = [SphericalHarmonics.cache(L_max, SphericalHarmonics.FullRange) for _ in 1:max_tasks]

    # BLAS combines the density weights with the real and imaginary parts of S separately. This lets
    # the real density matrices use real GEMMs.
    R_pos_real = zeros(T, n_densities, n_outputs)
    R_pos_imag = zeros(T, n_densities, n_outputs)
    typed_half = T(0.5)

    # Work through the Cartesian pairs in blocks to reduce memory usage.
    for pair_start in 1:pair_block:n_pairs
        # Catch the last incomplete block.
        pair_stop = min(pair_start + pair_block - 1, n_pairs)
        n_block = pair_stop - pair_start + 1

        # Restrict the reusable arrays to the active block and clear their previous values.
        real_block = @view response_real[:, 1:n_block]
        imag_block = @view response_imag[:, 1:n_block]
        fill!(real_block, zero(T))
        fill!(imag_block, zero(T))

        # Thread over the (i, j) pairs in this block. Each pair writes to its own column of S.
        n_tasks = chunk_count(n_block, nthreads())
        @sync for task_idx in 1:n_tasks
            Threads.@spawn begin
                # Get this task's buffers.
                gaussian = gaussian_pool[task_idx]
                jL = jL_pool[task_idx]
                jL_miller = jL_miller_pool[task_idx]
                Ylm_cache = Y_cache_pool[task_idx]

                for pair_in_block in chunk_range(task_idx, n_tasks, n_block)
                    # Convert the position in this block to the full pair list, then recover (i, j).
                    global_pair_idx = pair_start + pair_in_block - 1
                    i = pair_i[global_pair_idx]
                    j = pair_j[global_pair_idx]
                    ij_bin = W_tensor.ij_bins[full_bins[global_pair_idx]]

                    # Extract the geometry for this pair.
                    sigma_sq = sigma_ij[i, j] * sigma_ij[i, j]
                    R_mod = R_ij_mod[i, j]
                    theta_ij = R_ij_hat[i, j, 1]
                    phi_ij = R_ij_hat[i, j, 2]

                    # Precompute the spherical harmonics for this pair.
                    computePlmcostheta!(Ylm_cache, theta_ij, L_max)
                    computeYlm!(Ylm_cache, theta_ij, phi_ij, L_max)
                    Yvals = SphericalHarmonics.getY(Ylm_cache)

                    # Compute the Gaussian exp(-σ_{ij}²q²/2) and spherical Bessel functions
                    # j_L(qR_{ij}) across the q grid.
                    @inbounds for q_idx in 1:n_q
                        gaussian[q_idx] = exp(-typed_half * sigma_sq * q_powers[3, q_idx])
                        fill_spherical_bessel_column!(jL, jL_miller, q_idx,
                                                      q_grid_invA[q_idx] * R_mod, L_max)
                    end

                    # Loop over all non-zero W entries belonging to this Cartesian pair.
                    @inbounds for W_idx in ij_bin
                        # Extract the W indices and value, then find the matching Gaunt bin.
                        lambda = W_tensor.lambda[W_idx]
                        mu = W_tensor.mu[W_idx]
                        n_idx = W_tensor.n[W_idx] + 1
                        W_val = W_tensor.W_values[W_idx]
                        bin_key = lambda_mu_key[lambda + 1, mu + lambda + 1]

                        # Loop over the matching Gaunt coefficients. The m < 0 coefficients follow
                        # from conjugation symmetry and are restored after combining with T.
                        for gaunt_idx in gaunt_array.lambda_mu_bins[bin_key]
                            m = gaunt_array.m[gaunt_idx]
                            m < 0 && continue

                            L = gaunt_array.L[gaunt_idx]
                            l = gaunt_array.l[gaunt_idx]

                            # Compute the corresponding spherical-harmonic index M = m - μ and the
                            # q-independent angular part of this contribution.
                            M = m - mu
                            angular = W_val * gaunt_coeffs[gaunt_idx] *
                                      conj(Complex{T}(Yvals[(L, M)])) * i_powers[L + 1]
                            angular_real = real(angular)
                            angular_imag = imag(angular)

                            # Combine the triangular (ℓ,m) key and q index into one index so that one
                            # Cartesian pair occupies one contiguous column.
                            key_pos = (l * (l + 1)) ÷ 2 + m + 1
                            output_base = (key_pos - 1) * n_q

                            # Accumulate this q-dependent contribution to S_{ℓm,ij}(q).
                            @simd for q_idx in 1:n_q
                                radial = gaussian[q_idx] * q_powers[n_idx, q_idx] * jL[L + 1, q_idx]
                                output_idx = output_base + q_idx
                                real_block[output_idx, pair_in_block] += angular_real * radial
                                imag_block[output_idx, pair_in_block] += angular_imag * radial
                            end
                        end
                    end
                end
            end
        end

        # Pick the right part of density weights (T_ij combinations) to fill R.
        density_block = @view pair_density_weights[:, pair_start:pair_stop]

        # Combine every density-like matrix with the completed block of S coefficients.
        # The syntax is mul(C, A, B, alpha, beta) does C = α * A * B + β * C, so this adds A * B to C.
        mul!(R_pos_real, density_block, transpose(real_block), one(T), one(T))
        mul!(R_pos_imag, density_block, transpose(imag_block), one(T), one(T))
    end

    # Allocate the final full-m tensor and apply the common plane-wave and spin prefactor while copying
    # over the directly evaluated m ≥ 0 coefficients.
    R_tensor = Array{Complex{T}}(undef, n_densities, n_q, (l_max + 1)^2)
    scale = T(prefactor)
    @inbounds for l in 0:l_max
        full_key_base = l * l + l + 1
        pos_key_base = (l * (l + 1)) ÷ 2 + 1
        for m in 0:l
            pos_key = pos_key_base + m
            full_key = full_key_base + m
            sign = fast_neg1_pow(m - l, T) # (-1)^(m-l).
            output_base = (pos_key - 1) * n_q

            for q_idx in 1:n_q
                output_idx = output_base + q_idx
                @simd for density_idx in 1:n_densities
                    value = scale * Complex{T}(R_pos_real[density_idx, output_idx],
                                               R_pos_imag[density_idx, output_idx])
                    R_tensor[density_idx, q_idx, full_key] = value

                    # Fill in the m < 0 for free.
                    if m > 0
                        R_tensor[density_idx, q_idx, full_key_base - m] = sign * conj(value)
                    end
                end
            end
        end
    end
    return R_tensor
end

end
