# This module contains functions to contract the angular and radial parts of the form factor.

module ContractSphericalGrid

export contract_spherical_grid

using SphericalHarmonics
using Base.Threads
using LinearAlgebra: mul!, BLAS

using ...ThreadChunks: chunk_count, chunk_range

function contract_spherical_grid(R_tensor::Array{Complex{T}, 3}, theta_grid::Vector{T}, phi_grid::Vector{T})::Array{Complex{T}, 4} where {T<:AbstractFloat}
    """
    Contract the radial and angular parts of the form factor to obtain f_s(q, θ, ϕ) on a spherical grid, defined by:

        f_s(q, θ, ϕ) = ∑_{ℓ=0}^{l_max} ∑_{m=-ℓ}^{ℓ} R_{ℓm}(q) Y_ℓ^m(θ, ϕ),

    where R is the R tensor that depends only on the modulus of the momentum transfer, q,
    and Y_ℓ^m are the spherical harmonics.

    Spherical harmonics are computed once per (θ, ϕ) point and reused for all transitions.

    # Arguments:
    - R_lm::Array{Complex{T}, 3}: The batched R tensor with dimensions (n_transitions, n_q, n_keys), keyed by key(ℓ,m) = ℓ^2 + (ℓ + m) + 1.
    - theta_grid::Vector{T}: The θ values at which to evaluate the form factor, in radians.
    - phi_grid::Vector{T}: The ϕ values at which to evaluate the form factor, in radians.

    # Returns:
    - f_s::Array{Complex{T}, 4}: The spherical form factor with dimensions (n_transitions, n_q, n_θ, n_ϕ).
    """

    # Get the grid sizes.
    n_transitions = size(R_tensor, 1)
    n_q = size(R_tensor, 2)
    n_theta = length(theta_grid)
    n_phi = length(phi_grid)
    n_keys = size(R_tensor, 3)

    # Determine l_max from the R tensor dimensions.
    # The total number of keys is (ℓ_max + 1)^2.
    l_max = round(Int, sqrt(n_keys)) - 1

    # Allocate the output form factor array.
    f_s = Array{Complex{T}}(undef, n_transitions, n_q, n_theta, n_phi)

    # Reshape the R tensor to a matrix so we can use a BLAS GEMV.
    R_matrix = reshape(R_tensor, n_transitions * n_q, n_keys)

    # Now loop over the angular grid and construct the form factor.
    n_chunks = chunk_count(n_theta, nthreads())

    # Avoid BLAS oversubscription, since each task below makes its own BLAS calls.
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try

    @sync for chunk in 1:n_chunks
        Threads.@spawn begin
            # Each task owns its spherical harmonic cache, to avoid races.
            Y_cache = SphericalHarmonics.cache(l_max, SphericalHarmonics.FullRange)
            harmonics = Vector{Complex{T}}(undef, n_keys)

            for theta_idx in chunk_range(chunk, n_chunks, n_theta)
                # Enumerate doesn't work with threads, so we grab θ manually.
                theta = theta_grid[theta_idx]

                # Precompute associated Legendre polynomials for this θ.
                computePlmcostheta!(Y_cache, theta, l_max)

                for (phi_idx, phi) in enumerate(phi_grid)
                    # Compute spherical harmonics for this (θ, ϕ) once for all transitions.
                    computeYlm!(Y_cache, theta, phi, l_max)
                    Yvals = SphericalHarmonics.getY(Y_cache)

                    # Copy the harmonics into a flat vector, so we can use BLAS later to contract.
                    @inbounds for l in 0:l_max
                        # Precompute the key base.
                        key_base = l * l + l + 1
                        for m in -l:l
                            # Add the m offset.
                            harmonics[key_base + m] = Complex{T}(Yvals[(l, m)])
                        end
                    end

                    # BLAS is king.
                    mul!(vec(view(f_s, :, :, theta_idx, phi_idx)), R_matrix, harmonics)
                end
            end
        end
    end

    # Reset the BLAS thread count.
    finally
        BLAS.set_num_threads(blas_threads)
    end

    return f_s
end
end
