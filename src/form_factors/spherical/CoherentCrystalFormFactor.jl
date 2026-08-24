# This module builds the coherent unit-cell form factor for crystal excitations, by mixing the
# rotated monomer form factors with the Bloch eigenvector coefficients.

module CoherentCrystalFormFactor

using StaticArrays
using Quaternionic
using SphericalHarmonics
using VectorSpaceDarkMatter
using Base.Threads
using LinearAlgebra: mul!, dot, BLAS

using ..CrystalLattice: CrystalLatticeData, fold_to_bz
using ..Ewald: EwaldLongRangeData
using ..BlochHamiltonian: CrystalImage, CrystalExcitationBasis, BlochEigensystem, CrystalCouplings, solve_bloch_hamiltonian!
using ..ProjectFLM: build_projection_matrices, build_U_blocks, project_block!
using ..WignerRotations: wigner_d_blocks
using ..CrystalSymmetry: Stars, SymmetryOperation
using ...ThreadChunks: chunk_count, chunk_range

const VSDM = VectorSpaceDarkMatter

export rotate_R_tensors, compute_coherent_crystal_form_factor, compute_coherent_crystal_f_lm

function rotate_R_tensors(
        conformer_R_tensors::Vector{Array{Complex{T}, 3}},
        images::Vector{CrystalImage{T}},
        basis::CrystalExcitationBasis{T},
        l_max::Int,
    )::Array{Complex{T}, 3} where {T<:AbstractFloat}
    """
    Rotate each conformer's R tensor into the orientation of each image in the unit cell, producing

        f̄^{(A_i,s)}_{lμ}(q) = |R_{A,i}|^l Σ_m D^{(l)}_{μm}(R̃_{A,i}) f^{(A,s)}_{lm}(q).

    Pushing the rotation onto the coefficients like this is what keeps every image on the same
    unrotated q grid, so the spherical harmonics can be evaluated once and shared.

    # Arguments:
    - conformer_R_tensors::Vector{Array{Complex{T}, 3}}: R tensor per conformer, each with dimensions (n_transitions, n_q, n_keys).
    - images::Vector{CrystalImage{T}}: The molecules in the unit cell.
    - basis::CrystalExcitationBasis{T}: The localised excitation basis defining the λ ordering.
    - l_max::Int: The maximum angular momentum mode.

    # Returns:
    - Array{Complex{T}, 3}: The rotated coefficients with dimensions (n_lambda, n_q, n_keys).
    """

    n_lambda = length(basis.energies)
    n_q = size(conformer_R_tensors[1], 2)
    n_keys = size(conformer_R_tensors[1], 3)

    (l_max + 1)^2 == n_keys ||
        error("The R tensor has $(n_keys) keys, which does not match l_max = $(l_max) (expected $((l_max + 1)^2)).")

    rotated = Array{Complex{T}, 3}(undef, n_lambda, n_q, n_keys)

    # The D blocks depend only on the image, so build them once per image rather than per λ.
    image_blocks = [wigner_d_blocks(image.rotation, l_max) for image in images]

    # Thread over λ, the combined conformer + transition index.
    n_chunks = chunk_count(n_lambda, nthreads())

    @sync for chunk in 1:n_chunks
        Threads.@spawn begin
            # Each task owns its gather and accumulation buffers for a single l block.
            input_buffer = Vector{Complex{T}}(undef, 2 * l_max + 1)
            rotated_buffer = Vector{Complex{T}}(undef, 2 * l_max + 1)

            for lambda in chunk_range(chunk, n_chunks, n_lambda)
                image = images[basis.image_of[lambda]]
                blocks = image_blocks[basis.image_of[lambda]]
                R_tensor = conformer_R_tensors[image.conformer_index]
                transition_idx = basis.transition_of[lambda]

                for l in 0:l_max
                    key_start = l * l + 1
                    n_m = 2 * l + 1
                    D_l = blocks[l + 1]

                    # For improper rotations the odd-l blocks pick up a sign, unified as det(R)^l.
                    parity = Complex{T}(image.det_rotation^l)

                    input_view = @view input_buffer[1:n_m]
                    rotated_view = @view rotated_buffer[1:n_m]

                    for q_idx in 1:n_q
                        # Gather into a contiguous buffer so the matrix-vector product is fast.
                        @inbounds for key_offset in 1:n_m
                            input_view[key_offset] = R_tensor[transition_idx, q_idx, key_start + key_offset - 1]
                        end

                        mul!(rotated_view, D_l, input_view)

                        @inbounds for key_offset in 1:n_m
                            rotated[lambda, q_idx, key_start + key_offset - 1] = parity * rotated_view[key_offset]
                        end
                    end
                end
            end
        end
    end

    return rotated
end

function build_lambda_permutations(
        basis::CrystalExcitationBasis{T},
        operations::Vector{SymmetryOperation{T}},
    )::Vector{Vector{Int}} where {T<:AbstractFloat}
    """
    Builds the molecule-to-molecule map, λ -> Λ(λ), for each crystal symmetry operation.

    # Arguments:
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - operations::Vector{SymmetryOperation{T}}: The symmetry operations.

    # Returns:
    - Vector{Vector{Int}}: One permutation of 1:n_lambda per operation.
    """

    # Allocate the array for, and build the map (A_i, s) -> λ.
    n_lambda = length(basis.energies)
    n_images = length(basis.images)

    lambda_of = zeros(Int, n_images, basis.n_transitions)
    for lambda in 1:n_lambda
        lambda_of[basis.image_of[lambda], basis.transition_of[lambda]] = lambda
    end

    # Now build λ -> Λ(λ).
    permutations = Vector{Vector{Int}}(undef, length(operations))
    for (operation_idx, operation) in enumerate(operations)
        permutation = Vector{Int}(undef, n_lambda)

        for lambda in 1:n_lambda
            # Split λ back into A_i and s, the image and transition.
            image = basis.image_of[lambda]
            transition = basis.transition_of[lambda]

            # Get A_i -> Λ(A_i).
            mapped_image = operation.image_of[image]
            # Use that to build the λ -> Λ(λ) map.
            permutation[lambda] = lambda_of[mapped_image, transition]
        end

        permutations[operation_idx] = permutation
    end

    return permutations
end

function compute_coherent_crystal_form_factor(
        rotated_R::Array{Complex{T}, 3},
        basis::CrystalExcitationBasis{T},
        lattice::CrystalLatticeData{T},
        q_grid_invA::Vector{T},
        theta_grid::Vector{T},
        phi_grid::Vector{T},
        l_max::Int,
        stars::Stars{T},
        q_indices::UnitRange{Int} = 1:length(q_grid_invA);
        need_grid::Bool = false,
        couplings::Union{Nothing, CrystalCouplings{T}} = nothing,
        long_range::Union{Nothing, EwaldLongRangeData{T}} = nothing,
    ) where {T<:AbstractFloat}
    """
    Build the coherent unit-cell form factor for each crystal excitation Ψ,

        f_{Ψ,uc}(q) = Σ_{A,i,s} C*_{A_i,s}(k) exp(i q . τ_{A,i}) Σ_{l,μ} f̄^{(A_i,s)}_{lμ}(q) Y_l^μ(q̂),

    evaluated at q = k + G, and return |f_{Ψ,uc}(q)|² together with summary statistics of the band
    energies E_Ψ(k).

    The Brillouin zone wavevector k is a function of the full vector q, so in principle the Bloch
    Hamiltonian has to be rebuilt and diagonalised at every grid point. In practice, we use time
    reversal and the crystal symmetries to reduce the number of diagonalisations.

    Using time reversal symmetry, the eigenvalues and eigenvectors of H(k) and H(-k) are related by:

        E_Ψ(-k) = E_Ψ(k),   C_{A_i,s}(-k) = conj(C_{A_i,s}(k)).

    The crystal symmetries relate k-points on the "star", k -> k_Λ, by

        k_Λ = R_Λ^{-1} k,   E_Ψ(k_Λ) = E_Ψ(k),   C_{Λ(A_i),s}(k_Λ) = C_{A_i,s}(k) exp(-i k_Λ · L_{Λ,A_i}),
    
    with L_{Λ,A_i} = R_Λ τ_{A,i} - τ_{Λ(A_i)} + τ_Λ. Internally, time reversal is treated as a special case of a symmetry operation, so the same code handles both.

    # Arguments:
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients f̄, with dimensions (n_lambda, n_q, n_keys).
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - lattice::CrystalLatticeData{T}: The crystal lattice.
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å.
    - theta_grid::Vector{T}: The θ grid, in radians.
    - phi_grid::Vector{T}: The ϕ grid, in radians.
    - l_max::Int: The maximum angular momentum mode.
    - stars::Stars{T}: The direction stars and the operations relating them.
    - q_indices::UnitRange{Int}: Which q points to evaluate (default: all of them).
    - need_grid::Bool: Whether to also return the unsquared complex form factor (default: false).
    - couplings::Union{Nothing, CrystalCouplings{T}}: The "short-range" J_{λλ'}(ΔR), or nothing for no coupling.
    - long_range::Union{Nothing, EwaldLongRangeData{T}}: The Ewald long-range data, or nothing.

    # Returns:
    - f_sq::Array{T, 4}: |f_{Ψ,uc}(q)|² with dimensions (n_states, length(q_indices), n_theta, n_phi).
    - band_stats::Matrix{T}: Per-state band energy statistics in eV over this block, with columns
      (min, max, sum). The full block can be recovered from the Hamiltonian later if needed.
    - f_s::Union{Array{Complex{T}, 4}, Nothing}: The unsquared f_{Ψ,uc}(q) if need_grid is set, with
      the same dimensions as f_sq, otherwise nothing.
    """

    # Get the dimensions.
    n_lambda = length(basis.energies)
    n_theta = length(theta_grid)
    n_phi = length(phi_grid)
    n_q_block = length(q_indices)
    n_stars = length(stars.irreducible_directions)

    checkbounds(Bool, q_grid_invA, q_indices) ||
        error("The requested q indices $(q_indices) fall outside the q grid of length $(length(q_grid_invA)).")

    # Allocate the output arrays. The complex form factor is only kept when asked for, since it is
    # twice the size of the squared one and is only wanted for plotting.
    f_sq = Array{T, 4}(undef, n_lambda, n_q_block, n_theta, n_phi)
    f_s = need_grid ? Array{Complex{T}, 4}(undef, n_lambda, n_q_block, n_theta, n_phi) : nothing

    # Build the arrays τ(A_i) and Λ(λ).
    image_translations = [image.translation for image in basis.images]
    n_images = length(image_translations)
    image_of_lambda = basis.image_of
    permutations = build_lambda_permutations(basis, stars.operations)

    # Thread over stars, since each star is independent and the number of stars is usually very large.
    n_chunks = chunk_count(n_stars, nthreads())


    # Per-task band energy statistics, keyed by chunk so each has a single writer.
    chunk_min = [fill(T(Inf), n_lambda) for _ in 1:n_chunks]
    chunk_max = [fill(T(-Inf), n_lambda) for _ in 1:n_chunks]
    chunk_sum = [zeros(T, n_lambda) for _ in 1:n_chunks]

    @sync for chunk in 1:n_chunks
        Threads.@spawn begin
            # Give each task its own caches.
            Ylm_cache = SphericalHarmonics.cache(l_max, SphericalHarmonics.FullRange)
            eigensystem = BlochEigensystem(basis, long_range;
                n_cells = couplings === nothing ? 0 : length(couplings.cell_vectors))
            monomer_amplitude = Vector{Complex{T}}(undef, n_lambda)
            phased_amplitude = Vector{Complex{T}}(undef, n_lambda)
            image_phases = Vector{Complex{T}}(undef, n_images)

            # Allocate reusable buffers for each star.
            star_energies = Matrix{T}(undef, n_lambda, n_q_block)
            star_coefficients = Array{Complex{T}, 3}(undef, n_lambda, n_lambda, n_q_block) # These are C[λ, Ψ, q] = C^Ψ_λ(k). Stored for all q.
            star_wavevectors = Vector{SVector{3, T}}(undef, n_q_block)
            member_coefficients = Matrix{Complex{T}}(undef, n_lambda, n_lambda) # These are C[Λ(λ), Ψ] = C^Ψ_Λ(λ)(k_Λ), only stored at each q.

            local_min = chunk_min[chunk]
            local_max = chunk_max[chunk]
            local_sum = chunk_sum[chunk]

            # Thread over stars.
            for star_idx in chunk_range(chunk, n_chunks, n_stars)
                representative = stars.irreducible_directions[star_idx]

                # One diagonalisation per q point, for the whole star.
                for (q_local, q_idx) in enumerate(q_indices)
                    # Find the 1BZ wavevector for this q point.
                    k_vector, _ = fold_to_bz(lattice, q_grid_invA[q_idx] * representative)
                    star_wavevectors[q_local] = k_vector
                    # Diagonalise the Hamiltonian.
                    solve_bloch_hamiltonian!(eigensystem, basis, k_vector, couplings, long_range)
                    @inbounds for state_idx in 1:n_lambda
                        # Store the band energies and coefficients.
                        star_energies[state_idx, q_local] = eigensystem.energies[state_idx]
                        for lambda in 1:n_lambda
                            star_coefficients[lambda, state_idx, q_local] =
                                eigensystem.coefficients[lambda, state_idx]
                        end
                    end
                end

                for member in stars.members[star_idx]
                    operation = stars.operations[member.operation]
                    permutation = permutations[member.operation]
                    direction = member.direction

                    # k_Λ = R_Λ^{-1} k, with a further sign when the operation carries time reversal.
                    inverse_rotation = transpose(operation.rotation)
                    reversal = operation.conjugate ? -one(T) : one(T) # -1 for time reversal.

                    # Several (θ, ϕ) indices can point in the same direction, ̂n. For examople: the two poles
                    # have the same ϕ, whilst in general ϕ = 0 and 2π are the same direction.
                    # This is important because the spherical harmonics are only evaluated once per unique direction.
                    theta_idx, phi_idx = member.grid_indices[1]
                    theta = theta_grid[theta_idx]
                    phi = phi_grid[phi_idx]
                    computePlmcostheta!(Ylm_cache, theta, l_max)
                    
                    # Compute the spherical harmonics for this direction, which are shared across all q and λ.
                    computeYlm!(Ylm_cache, theta, phi, l_max)
                    Yvals = SphericalHarmonics.getY(Ylm_cache)

                    for (q_local, q_idx) in enumerate(q_indices)
                        q_vector = q_grid_invA[q_idx] * direction

                        # Contract the rotated coefficients with the shared spherical harmonics to
                        # get each localised molecular amplitude at this q.
                        @inbounds for lambda in 1:n_lambda
                            amplitude = zero(Complex{T})
                            for l in 0:l_max
                                # Convert l to the key index, up to the factor of m.
                                key_base = l * l + l + 1
                                for m in -l:l
                                    # Form f_lm * Y_lm, the form factor (per molecule and transition) at this grid point.
                                    amplitude += rotated_R[lambda, q_idx, key_base + m] * Complex{T}(Yvals[(l, m)])
                                end
                            end
                            monomer_amplitude[lambda] = amplitude
                        end

                        # Apply the exp(i q . τ) phase that places each molecule in the cell.
                        # First compute the phase for each image.
                        @inbounds for image in 1:n_images
                            image_phases[image] = cis(dot(q_vector, image_translations[image]))
                        end
                        # Then apply it to each λ = (A_i, s).
                        @inbounds for lambda in 1:n_lambda
                            phased_amplitude[lambda] =
                                monomer_amplitude[lambda] * image_phases[image_of_lambda[lambda]]
                        end

                        # Now compute the eigenvectors for the star.
                        k_lambda = reversal * (inverse_rotation * star_wavevectors[q_local])
                        @inbounds for lambda in 1:n_lambda
                            # Compute e^{-i k_Λ . L_{Λ,A_i}} that rephases eigenvectors/mixing coefficients.
                            phase_star = cis(-dot(k_lambda, operation.offsets[image_of_lambda[lambda]]))
                            target = permutation[lambda]
                            # Handle time reversal with conjugation if necessary.
                            if operation.conjugate
                                for state_idx in 1:n_lambda
                                    member_coefficients[target, state_idx] =
                                        conj(star_coefficients[lambda, state_idx, q_local]) * phase_star
                                end
                            else
                                for state_idx in 1:n_lambda
                                    member_coefficients[target, state_idx] =
                                        star_coefficients[lambda, state_idx, q_local] * phase_star
                                end
                            end
                        end

                        # Compute the coherent form factor at this q.
                        @inbounds for state_idx in 1:n_lambda
                            total = zero(Complex{T})
                            for lambda in 1:n_lambda
                                total += conj(member_coefficients[lambda, state_idx]) * phased_amplitude[lambda]
                            end

                            energy = star_energies[state_idx, q_local]
                            magnitude = abs2(total)
                            # Write the squared form factor, now at all points, not just unique directions.
                            for (write_theta, write_phi) in member.grid_indices
                                f_sq[state_idx, q_local, write_theta, write_phi] = magnitude
                                if f_s !== nothing
                                    f_s[state_idx, q_local, write_theta, write_phi] = total
                                end
                                
                                # Track the band statistics for each chunk.
                                local_min[state_idx] = min(local_min[state_idx], energy)
                                local_max[state_idx] = max(local_max[state_idx], energy)
                                local_sum[state_idx] += energy
                            end
                        end
                    end
                end
            end
        end
    end

    # Merge the per-task band statistics.
    band_stats = Matrix{T}(undef, n_lambda, 3)
    @inbounds for state_idx in 1:n_lambda
        band_stats[state_idx, 1] = minimum(chunk_min[chunk][state_idx] for chunk in 1:n_chunks)
        band_stats[state_idx, 2] = maximum(chunk_max[chunk][state_idx] for chunk in 1:n_chunks)
        band_stats[state_idx, 3] = sum(chunk_sum[chunk][state_idx] for chunk in 1:n_chunks)
    end

    return f_sq, band_stats, f_s
end

function compute_coherent_crystal_f_lm(
        rotated_R::Array{Complex{T}, 3},
        basis::CrystalExcitationBasis{T},
        lattice::CrystalLatticeData{T},
        q_grid_invA::Vector{T},
        theta_grid::Vector{T},
        phi_grid::Vector{T},
        l_max::Int,
        stars::Stars{T};
        q_block::Int = 8,
        need_grid::Bool = false,
        couplings::Union{Nothing, CrystalCouplings{T}} = nothing,
        long_range::Union{Nothing, EwaldLongRangeData{T}} = nothing,
    ) where {T<:AbstractFloat}
    """
    Compute the coherent crystal form factor and project it onto real spherical harmonics,
    streaming over q so that the full squared form factor is never held in memory at once.

    # Arguments:
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients f̄, with dimensions (n_lambda, n_q, n_keys).
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - lattice::CrystalLatticeData{T}: The crystal lattice.
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å.
    - theta_grid::Vector{T}: The θ grid, in radians.
    - phi_grid::Vector{T}: The ϕ grid, in radians.
    - l_max::Int: The maximum angular momentum mode.
    - stars::Stars{T}: The direction stars and the operations relating them.
    - q_block::Int: How many q points to process at a time (default: 8).
    - need_grid::Bool: Whether to also assemble the unsquared complex form factor on the full grid
      (default: false).
    - couplings::Union{Nothing, CrystalCouplings{T}}: The J_{λλ'}(ΔR), or nothing for no coupling.
    - long_range::Union{Nothing, EwaldLongRangeData{T}}: The Ewald long-range data, or nothing.

    # Returns:
    - f_lm::Array{T, 3}: The real spherical harmonic coefficients of |f_{Ψ,uc}|², with dimensions (n_states, n_q, n_keys).
    - band_summary::Matrix{T}: Per-state band energy statistics in eV, with columns (min, max, mean)
      taken over the whole q grid. With the intermolecular couplings switched off these collapse to
      the monomer energies with zero bandwidth, which is itself a useful check.
    - f_s::Union{Array{Complex{T}, 4}, Nothing}: The unsquared f_{Ψ,uc}(q) on the full grid if
      need_grid is set, with dimensions (n_states, n_q, n_theta, n_phi), otherwise nothing.
    """

    n_states = length(basis.energies)
    n_q = length(q_grid_invA)
    n_theta = length(theta_grid)
    n_phi = length(phi_grid)
    n_keys = (l_max + 1)^2

    # The projection matrices and the complex-to-real transformation are shared across all blocks.
    A_real, A_imag = build_projection_matrices(theta_grid, phi_grid, l_max)
    U_blocks = build_U_blocks(l_max, T)

    f_lm = Array{T, 3}(undef, n_states, n_q, n_keys)
    f_s = need_grid ? Array{Complex{T}, 4}(undef, n_states, n_q, n_theta, n_phi) : nothing

    # Running band statistics, accumulated across blocks.
    band_min = fill(T(Inf), n_states)
    band_max = fill(T(-Inf), n_states)
    band_sum = zeros(T, n_states)
    band_count = 0

    # Avoid BLAS oversubscription.
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try

    for q_start in 1:q_block:n_q
        q_stop = min(q_start + q_block - 1, n_q)
        q_indices = q_start:q_stop

        # Compute the coherent crystal form factor for this block of q, streaming over θ and ϕ so the
        # full |f|² grid is never held in memory at once.
        f_sq_block, block_stats, f_s_block = compute_coherent_crystal_form_factor(
            rotated_R, basis, lattice, q_grid_invA, theta_grid, phi_grid, l_max, stars, q_indices;
            need_grid = need_grid, couplings = couplings, long_range = long_range)

        project_block!(f_lm, f_sq_block, A_real, A_imag, U_blocks, q_start, l_max)

        # Store the unsquared form factor on the full grid if requested, so it can be plotted later.
        if f_s !== nothing
            f_s[:, q_indices, :, :] .= f_s_block
        end

        # Update the running band statistics.
        @inbounds for state_idx in 1:n_states
            band_min[state_idx] = min(band_min[state_idx], block_stats[state_idx, 1])
            band_max[state_idx] = max(band_max[state_idx], block_stats[state_idx, 2])
            band_sum[state_idx] += block_stats[state_idx, 3]
        end
        band_count += length(q_indices) * length(theta_grid) * length(phi_grid)
    end

    # Reset the BLAS thread count to its original value.
    finally
        BLAS.set_num_threads(blas_threads)
    end

    # Compile the band statistics into a single matrix for output, and compute the mean energy per state.
    band_summary = Matrix{T}(undef, n_states, 3)
    @inbounds for state_idx in 1:n_states
        band_summary[state_idx, 1] = band_min[state_idx]
        band_summary[state_idx, 2] = band_max[state_idx]
        band_summary[state_idx, 3] = band_sum[state_idx] / band_count
    end

    return f_lm, band_summary, f_s
end

end
