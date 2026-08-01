# This module builds the coherent unit-cell form factor for crystal excitations, by mixing the
# rotated monomer form factors with the Bloch eigenvector coefficients.

module CoherentCrystalFormFactor

using StaticArrays
using Quaternionic
using SphericalHarmonics
using VectorSpaceDarkMatter
using Base.Threads
using LinearAlgebra: mul!, dot

using ..CrystalLattice: CrystalLatticeData, fold_to_bz
using ..BlochHamiltonian: CrystalImage, CrystalExcitationBasis, BlochEigensystem, solve_bloch_hamiltonian!
using ..ProjectFLM: build_projection_matrices, build_U_blocks, project_block!
using ...ThreadChunks: chunk_count, chunk_range

const VSDM = VectorSpaceDarkMatter

# SphericalFunctions.jl stores its Wigner D matrices transposed and/or conjugated relative to its own
# documentation, so we follow the same load-time probe VSDM uses rather than hardcoding an assumption.
const D_NEEDS_TRANSPOSE = VSDM.do_transpose
const D_NEEDS_CONJUGATE = VSDM.do_conjugate

export rotate_R_tensors, compute_coherent_crystal_form_factor, compute_coherent_crystal_f_lm

function wigner_d_blocks(rotation::Quaternionic.Rotor{T}, l_max::Int)::Vector{Matrix{Complex{T}}} where {T<:AbstractFloat}
    """
    Build the Wigner D matrix blocks that rotate spherical harmonic coefficients, in the convention

        Y_l^m(R̃^{-1} q̂) = Σ_μ D^{(l)}_{μm}(R̃) Y_l^μ(q̂).

    VSDM stores its D coefficients flat, so we extract the (2l+1)x(2l+1) block for each l and apply
    whichever of transpose and conjugate the probe above calls for, which puts them in the [μ, m]
    orientation this convention needs.

    # Arguments:
    - rotation::Quaternionic.Rotor{T}: The proper rotation R̃ as a unit quaternion rotor.
    - l_max::Int: The maximum angular momentum mode.

    # Returns:
    - Vector{Matrix{Complex{T}}}: The D block for each l, indexed [μ + l + 1, m + l + 1].
    """

    D_buffer = VSDM.D_prep(l_max)
    VSDM.D_matrices!(D_buffer, rotation)
    D_values = D_buffer[1]

    blocks = Vector{Matrix{Complex{T}}}(undef, l_max + 1)
    for l in 0:l_max
        block_start = VSDM.WignerDindex(l, -l, -l)
        block_stop = VSDM.WignerDindex(l, l, l)
        block = reshape(collect(D_values[block_start:block_stop]), 2 * l + 1, 2 * l + 1)
        # Rotate if needed, as per VSDM.
        D_NEEDS_TRANSPOSE && (block = permutedims(block))
        D_NEEDS_CONJUGATE && (block = conj(block))
        blocks[l + 1] = Matrix{Complex{T}}(block)
    end

    return blocks
end

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

function compute_coherent_crystal_form_factor(
        rotated_R::Array{Complex{T}, 3},
        basis::CrystalExcitationBasis{T},
        lattice::CrystalLatticeData{T},
        q_grid_invA::Vector{T},
        theta_grid::Vector{T},
        phi_grid::Vector{T},
        l_max::Int,
        q_indices::UnitRange{Int} = 1:length(q_grid_invA);
        need_grid::Bool = false,
    ) where {T<:AbstractFloat}
    """
    Build the coherent unit-cell form factor for each crystal excitation Ψ,

        f_{Ψ,uc}(q) = Σ_{A,i,s} C*_{A_i,s}(k) exp(i q . τ_{A,i}) Σ_{l,μ} f̄^{(A_i,s)}_{lμ}(q) Y_l^μ(q̂),

    evaluated at q = k + G, and return |f_{Ψ,uc}(q)|² together with summary statistics of the band
    energies E_Ψ(k).

    The Brillouin zone wavevector k is a function of the full vector q, so the Bloch Hamiltonian is
    rebuilt and diagonalised at every grid point.

    # Arguments:
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients f̄, with dimensions (n_lambda, n_q, n_keys).
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - lattice::CrystalLatticeData{T}: The crystal lattice.
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å.
    - theta_grid::Vector{T}: The θ grid, in radians.
    - phi_grid::Vector{T}: The ϕ grid, in radians.
    - l_max::Int: The maximum angular momentum mode.
    - q_indices::UnitRange{Int}: Which q points to evaluate, so the caller can stream over q rather
      than materialising the whole grid at once (default: all of them).
    - need_grid::Bool: Whether to also return the unsquared complex form factor (default: false).

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

    checkbounds(Bool, q_grid_invA, q_indices) ||
        error("The requested q indices $(q_indices) fall outside the q grid of length $(length(q_grid_invA)).")

    # Allocate the output arrays. The complex form factor is only kept when asked for, since it is
    # twice the size of the squared one and is only wanted for plotting.
    f_sq = Array{T, 4}(undef, n_lambda, n_q_block, n_theta, n_phi)
    f_s = need_grid ? Array{Complex{T}, 4}(undef, n_lambda, n_q_block, n_theta, n_phi) : nothing

    # Precompute the translation vectors per λ, so the inner loop is a plain dot product.
    tau_of_lambda = [basis.images[basis.image_of[lambda]].translation for lambda in 1:n_lambda]

    # Thread over theta, which lets each task own its spherical harmonic cache and Bloch eigensystem.
    n_chunks = chunk_count(n_theta, nthreads())

    # Per-task band energy accumulators, keyed by chunk so each has a single writer.
    chunk_min = [fill(T(Inf), n_lambda) for _ in 1:n_chunks]
    chunk_max = [fill(T(-Inf), n_lambda) for _ in 1:n_chunks]
    chunk_sum = [zeros(T, n_lambda) for _ in 1:n_chunks]

    @sync for chunk in 1:n_chunks
        Threads.@spawn begin
            # Each task owns its scratch: harmonics cache, Bloch eigensystem, and the per-λ amplitudes.
            Ylm_cache = SphericalHarmonics.cache(l_max, SphericalHarmonics.FullRange)
            eigensystem = BlochEigensystem(basis)
            monomer_amplitude = Vector{Complex{T}}(undef, n_lambda)
            phased_amplitude = Vector{Complex{T}}(undef, n_lambda)

            local_min = chunk_min[chunk]
            local_max = chunk_max[chunk]
            local_sum = chunk_sum[chunk]

            for theta_idx in chunk_range(chunk, n_chunks, n_theta)
                theta = theta_grid[theta_idx]
                sin_theta, cos_theta = sincos(theta)

                computePlmcostheta!(Ylm_cache, theta, l_max)

                for (phi_idx, phi) in enumerate(phi_grid)
                    # Compute the spherical harmonics for this (θ, ϕ) pair, which are shared across all q and λ.
                    computeYlm!(Ylm_cache, theta, phi, l_max)
                    Yvals = SphericalHarmonics.getY(Ylm_cache)

                    sin_phi, cos_phi = sincos(phi)
                    direction = SVector{3, T}(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)

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
                        @inbounds for lambda in 1:n_lambda
                            phase_angle = dot(q_vector, tau_of_lambda[lambda])
                            phased_amplitude[lambda] = monomer_amplitude[lambda] * cis(phase_angle)
                        end

                        # q = k + G, so fold to get the Bloch wavevector and solve there.
                        k_vector, _ = fold_to_bz(lattice, q_vector)
                        solve_bloch_hamiltonian!(eigensystem, basis, k_vector)

                        # Mix the molecular amplitudes coherently with the eigenvector coefficients.
                        coefficients = eigensystem.coefficients
                        @inbounds for state_idx in 1:n_lambda
                            total = zero(Complex{T})
                            for lambda in 1:n_lambda
                                total += conj(coefficients[lambda, state_idx]) * phased_amplitude[lambda]
                            end
                            f_sq[state_idx, q_local, theta_idx, phi_idx] = abs2(total)
                            if f_s !== nothing
                                f_s[state_idx, q_local, theta_idx, phi_idx] = total
                            end

                            # Reduce the band energy as we go rather than storing the full grid.
                            energy = eigensystem.energies[state_idx]
                            local_min[state_idx] = min(local_min[state_idx], energy)
                            local_max[state_idx] = max(local_max[state_idx], energy)
                            local_sum[state_idx] += energy
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
        l_max::Int;
        q_block::Int = 8,
        need_grid::Bool = false,
    ) where {T<:AbstractFloat}
    """
    Compute the coherent crystal form factor and project it straight onto real spherical harmonics,
    streaming over q so that the full squared form factor is never held in memory at once.

    # Arguments:
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients f̄, with dimensions (n_lambda, n_q, n_keys).
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - lattice::CrystalLatticeData{T}: The crystal lattice.
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å.
    - theta_grid::Vector{T}: The θ grid, in radians.
    - phi_grid::Vector{T}: The ϕ grid, in radians.
    - l_max::Int: The maximum angular momentum mode.
    - q_block::Int: How many q points to process at a time (default: 8).
    - need_grid::Bool: Whether to also assemble the unsquared complex form factor on the full grid
      (default: false).

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

    for q_start in 1:q_block:n_q
        q_stop = min(q_start + q_block - 1, n_q)
        q_indices = q_start:q_stop

        # Compute the coherent crystal form factor for this block of q, streaming over θ and ϕ so the
        # full |f|² grid is never held in memory at once.
        f_sq_block, block_stats, f_s_block = compute_coherent_crystal_form_factor(
            rotated_R, basis, lattice, q_grid_invA, theta_grid, phi_grid, l_max, q_indices;
            need_grid = need_grid)

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
