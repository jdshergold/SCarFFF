# This module computes the Frenkel exciton couplings J between localised molecular excitations,
# summed over neighbouring unit cells.

module CouplingJ

using StaticArrays
using SphericalHarmonics
using Quaternionic
using Base.Threads
using LinearAlgebra: norm, cross, dot, mul!, normalize, BLAS

using ..CrystalLattice: CrystalLatticeData, ALPHA_EM, HBAR_C_EV_ANGSTROM
using ..Ewald: EwaldParameters, short_range_kernel
using ..BlochHamiltonian: CrystalExcitationBasis, CrystalCouplings
using ..ConstructRTensor: fill_spherical_bessel_column!
using ..ConstructFLMTensor: load_gaunt_array
using ..ProjectFLM: quadrature_weights
using ..WignerRotations: wigner_d_blocks
using ...FastPowers: fast_i_pow, fast_neg1_pow

export NeighbourCells, enumerate_neighbour_cells, compute_couplings, subtract_self_term!,
       image_translation_span

function image_translation_span(basis::CrystalExcitationBasis{T})::T where {T<:AbstractFloat}
    """
    The largest separation |τ_i - τ_j| between two molecules in the unit cell.

    A cutoff on the molecular separation d = τ_i - τ_j - ΔR is only guaranteed to be respected by a
    neighbour list built on |ΔR| if that list is padded by this much, since the translations can
    carry a pair either side of the lattice vector by up to this distance.

    # Arguments:
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.

    # Returns:
    - T: The span, in Å.
    """

    span = zero(T)
    for image_i in basis.images, image_j in basis.images
        span = max(span, norm(image_i.translation - image_j.translation))
    end

    return span
end

struct NeighbourCells{T<:AbstractFloat}
    """
    The lattice vectors ΔR included in the coupling sum, and the bookkeeping needed to relate a cell
    to its negative, ΔR → -ΔR.

    # Fields:
    - vectors::Vector{SVector{3, T}}: The lattice vectors ΔR, in Å.
    - indices::Vector{NTuple{3, Int}}: The integer coordinates of each ΔR in the lattice basis.
    - negative_of::Vector{Int}: For each cell, the index of the cell holding -ΔR.
    """
    vectors::Vector{SVector{3, T}}
    indices::Vector{NTuple{3, Int}}
    negative_of::Vector{Int}
end

function enumerate_neighbour_cells(lattice::CrystalLatticeData{T}, cutoff::Real)::NeighbourCells{T} where {T<:AbstractFloat}
    """
    Enumerate the lattice vectors ΔR with |ΔR| <= cutoff, including ΔR = 0.

    The search range along each axis is set from the cutoff and the spacing between lattice planes,
    which is the cell volume divided by the area of the opposite face. This ensures that we do not miss cells,
    which might happen using the naive distance cutoff/|a_i| along each axis.

    # Arguments:
    - lattice::CrystalLatticeData{T}: The crystal lattice.
    - cutoff::Real: The real-space cutoff, in Å.

    # Returns:
    - NeighbourCells{T}: The included cells and their negation map.
    """

    # Get the lattice vectors.
    a1 = SVector{3, T}(lattice.direct[1, :])
    a2 = SVector{3, T}(lattice.direct[2, :])
    a3 = SVector{3, T}(lattice.direct[3, :])

    # Compute the interplanar spacings, which are the cell volume divided by the area of the opposite face.
    spacings = (
        lattice.volume / norm(cross(a2, a3)),
        lattice.volume / norm(cross(a3, a1)),
        lattice.volume / norm(cross(a1, a2)),
    )

    # The number of steps along each axis needed to cover the cutoff.
    steps_needed = ntuple(i -> cutoff / spacings[i], 3)

    # Round that up, and take one cell of slack in case the division lands just below an integer.
    # This ensures that we do not miss cells. If any of these fall outside the cutoff, they will be filtered out below.
    search = ntuple(i -> Int(ceil(steps_needed[i])) + 1, 3)

    vectors = SVector{3, T}[]
    indices = NTuple{3, Int}[]
    # Now store only the cells that are within the cutoff, which is a sphere rather than a box.
    for n1 in -search[1]:search[1], n2 in -search[2]:search[2], n3 in -search[3]:search[3]
        vector = n1 * a1 + n2 * a2 + n3 * a3
        norm(vector) <= cutoff || continue
        push!(vectors, vector)
        push!(indices, (n1, n2, n3))
    end

    # Map each cell to its negative.
    index_lookup = Dict(idx => position for (position, idx) in enumerate(indices))
    negative_of = [index_lookup[(-idx[1], -idx[2], -idx[3])] for idx in indices]

    return NeighbourCells{T}(vectors, indices, negative_of)
end

function rotor_to_z(direction::SVector{3, T})::Quaternionic.Rotor{T} where {T<:AbstractFloat}
    """
    The shortest rotation carrying a unit vector onto ẑ, as a rotor.

    A short note on quaternions. A quaternion (w, x, y, z) in our case represents
    a rotation about the axis (x, y, z), normalised, by an angle θ = 2 arccos(w).

    # Arguments:
    - direction::SVector{3, T}: A unit vector.

    # Returns:
    - Quaternionic.Rotor{T}: The rotor R̃ with R̃ d̂ = ẑ.
    """

    # Get the cosine of the angle between the direction and z.
    cosine = clamp(direction[3], -one(T), one(T))

    # If the direction is already along z, return the idenity quaternion.
    cosine > 1 - eps(T)^(2//3) && return Quaternionic.rotor(T[1, 0, 0, 0])
    # If the direction is antiparallel, rotate 180 deg about any xy-plane-axis. We choose x.
    cosine < -1 + eps(T)^(2//3) && return Quaternionic.rotor(T[0, 1, 0, 0])

    # Otherwise, find the axis n = d x z and the angle θ = 2 arccos(w).
    axis = normalize(cross(direction, SVector{3, T}(0, 0, 1)))
    half = acos(cosine) / 2

    w = cos(half)
    # Multiply axis by sin(half) so the rotor has w^2 + |v|^2 = 1, with v = sin(half) * (x, y, z).
    x, y, z = sin(half) .* axis 

    return Quaternionic.rotor(T[w, x, y, z])
end

struct PairFrameKernel{T<:AbstractFloat}
    """
    The Gaunt data of the short-range kernel, reduced to what survives in the pair frame.
    This is the object:

        A^{αβ}_{L,l,l',μ} = i^L √((2L+1)/4π) G_{L0,lμ}^{l'μ},

    which is to be multplied by a bessel function of order L later, and summed over L.

    # Fields:
    - triples::Vector{NTuple{3, Int}}: The surviving (l, l', μ), sorted by destination key.
    - offsets::Vector{Int}: Start of each triple's entries in the two member arrays, plus a final stop.
    - member_L::Vector{Int}: The Bessel order L of each entry.
    - member_weight::Vector{Complex{T}}: (-1)^μ i^L √((2L+1)/4π) times the Gaunt coefficient.
    """
    triples::Vector{NTuple{3, Int}}
    offsets::Vector{Int}
    member_L::Vector{Int}
    member_weight::Vector{Complex{T}}
end

function build_pair_frame_kernel(gaunt_path::String, ::Type{T})::PairFrameKernel{T} where {T<:AbstractFloat}
    """
    Group the M = 0 Gaunt coefficients into the pair-frame kernel table.

    # Arguments:
    - gaunt_path::String: Path to the coupling-shape Gaunt coefficients.
    - T::Type: The float type.

    # Returns:
    - PairFrameKernel{T}: The reduced table.
    """

    gaunt = load_gaunt_array(gaunt_path, T)

    grouped = Dict{NTuple{3, Int}, Vector{Int}}()
    # Pick out only the M = 0 Gaunt coefficients.
    for idx in eachindex(gaunt.coefficients)
        iszero(gaunt.m[idx]) || continue
        key = (gaunt.lambda[idx], gaunt.L[idx], gaunt.mu[idx])
        # This is some shorthand. It means, if grouped[key] doesn't exist, create it as an empty Int vector, then push idx onto it.
        # If it does exist, just push idx onto it. Saves some lookups.
        push!(get!(grouped, key, Int[]), idx)
    end

    # Define the destination key function. Maps (l', μ) to l'^2 + (l' + μ) + 1.
    destination_key(triple) = triple[2] * triple[2] + (triple[2] + triple[3]) + 1
    # Sort the keys by the destination key. Speeds up memory access later.
    triples = sort!(collect(keys(grouped)); by = destination_key)

    # Now flatten the grouped data into the kernel table.
    offsets = Vector{Int}(undef, length(triples) + 1)
    member_L = Int[]
    member_weight = Complex{T}[]
    for (position, triple) in enumerate(triples)
        offsets[position] = length(member_L) + 1
        sign = fast_neg1_pow(triple[3], T)
        for idx in grouped[triple]
            L = gaunt.l[idx]
            push!(member_L, L)
            push!(member_weight,
                  sign * fast_i_pow(L, T) * sqrt((2 * L + 1) / (4 * T(π))) * gaunt.coefficients[idx])
        end
    end
    offsets[end] = length(member_L) + 1

    return PairFrameKernel{T}(triples, offsets, member_L, member_weight)
end

function compute_couplings(
        rotated_R::Array{Complex{T}, 3},
        basis::CrystalExcitationBasis{T},
        cells::NeighbourCells{T},
        q_grid_invA::Vector{T},
        l_max::Int,
        gaunt_path::String;
        parameters::Union{Nothing, EwaldParameters{T}} = nothing,
    )::CrystalCouplings{T} where {T<:AbstractFloat}
    """
    Compute the short-range Frenkel exciton couplings, in the frame of each pair αβ,

        J^SR_{A_i s, B_j t}(ΔR) = (2 α_EM / π) Σ_{ℓ,ℓ'} Σ_μ ∫dq κ(q)
                                      f̄^{(A_i,s)}_{ℓμ,αβ}(q) K^{∥,A_iB_j}_{ℓℓ',μ}(q) f̄^{(B_j,t)*}_{ℓ'μ,αβ}(q),

    where d_{αβ} = τ_λ - τ_λ' - ΔR is the vector between the two molecules, and

        K^{∥,A_iB_j}_{ℓℓ',μ}(q) = Σ_L i^L √((2L+1)/4π) j_L(q d_{αβ}) G_{L0,ℓμ}^{ℓ'μ},

        f̄^{(A_i,s)}_{ℓμ,αβ}(q)  = Σ_ν D_{μν}(S^{-1}_{αβ}) f̄^{(A_i,s)}_{ℓν}(q).

    The short-range Ewald kernel is κ(q) = 1 - exp(-q²/4η²), with η the Ewald parameter.

    # Arguments:
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients f̄, with dimensions (n_lambda, n_q, n_keys).
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - cells::NeighbourCells{T}: The lattice vectors to sum over.
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å.
    - l_max::Int: The maximum angular momentum mode of the form factor.
    - gaunt_path::String: Path to the coupling-shape Gaunt coefficients.
    - parameters::Union{Nothing, EwaldParameters{T}}: The Ewald split to use, or nothing for the bare
      Coulomb kernel and hence the full, only conditionally convergent, lattice sum.

    # Returns:
    - CrystalCouplings{T}: The couplings in eV, with their lattice vectors.
    """

    # Get the dimensions.
    n_lambda = length(basis.energies)
    n_q = length(q_grid_invA)
    n_cells = length(cells.vectors)
    n_keys = (l_max + 1)^2
    L_max = 2 * l_max

    # Check that the rotated coefficients have the right dimensions.
    (l_max + 1)^2 == size(rotated_R, 3) ||
        error("The rotated coefficients have $(size(rotated_R, 3)) keys, which does not match l_max = $(l_max).")

    # Build the pair-frame kernel, which is K up to the Bessel function.
    kernel = build_pair_frame_kernel(gaunt_path, T)
    n_triples = length(kernel.triples)

    radial_weights = quadrature_weights(q_grid_invA, false)
    # Multiply by the short-range kernel if using Ewald.
    parameters !== nothing && (radial_weights .*= short_range_kernel(q_grid_invA, parameters.eta))
    prefactor = Complex{T}(2 * ALPHA_EM / π * HBAR_C_EV_ANGSTROM)

    n_images = length(basis.images)
    # Find the lambdas for each image.
    lambdas_of_image = [findall(==(image_idx), basis.image_of) for image_idx in 1:n_images]
    # Every image carries the same transitions, which is what lets the buffers below be fixed size.
    n_transitions = basis.n_transitions
    all(==(n_transitions), length.(lambdas_of_image)) || error(
        "Each image should carry $(n_transitions) transitions, but the counts are " *
        "$(length.(lambdas_of_image))."
    )

    # Store the rotated flm (R tensor) coefficients for all molecules in the unit cell.
    # The first are the unconjugated, second are conjugated. We store them with different indices to speed up
    # matmuls later, by saving on transposing and conjugating.
    first_coefficients = [Array{Complex{T}, 3}(undef, length(lambdas_of_image[i]), n_q, n_keys)
                          for i in 1:n_images]
    second_coefficients = [Array{Complex{T}, 3}(undef, n_q, n_keys, length(lambdas_of_image[i]))
                           for i in 1:n_images]
    for image_idx in 1:n_images, (slot, lambda) in enumerate(lambdas_of_image[image_idx])
        @inbounds for key in 1:n_keys, q_idx in 1:n_q
            value = rotated_R[lambda, q_idx, key]
            first_coefficients[image_idx][slot, q_idx, key] = value
            second_coefficients[image_idx][q_idx, key, slot] = conj(value)
        end
    end

    # One job per (image pair, cell). Molecules do not interact with themselves, and under
    # the Ewald split the cutoff applies to the molecular separation, not to the lattice vector.
    jobs = NTuple{3, Int}[]
    for image_i in 1:n_images, image_j in image_i:n_images, cell_idx in 1:n_cells
        # Skip i = j and zero lattice vectors.
        image_i == image_j && all(iszero, cells.indices[cell_idx]) && continue
        # Build τ_i - τ_j - ΔR.
        separation = basis.images[image_i].translation - basis.images[image_j].translation -
                     cells.vectors[cell_idx]
        distance = norm(separation)
        distance > eps(T) || error(
            "Molecules $(image_i) and $(image_j) coincide at ΔR = $(cells.indices[cell_idx]), " *
            "which should be impossible. Check the translation vectors in the crystal metadata."
        )
        # Check that the molecule pair is within the cutoff.
        parameters !== nothing && distance > parameters.R_max && continue
        push!(jobs, (image_i, image_j, cell_idx))
    end

    couplings = zeros(Complex{T}, n_cells, n_lambda, n_lambda)

    # Avoid BLAS oversubscription.
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        n_chunks = min(nthreads(), max(length(jobs), 1))
        @sync for chunk in 1:n_chunks
            Threads.@spawn begin
                # Allocate buffers for each thread.
                bessel_table = Array{T, 2}(undef, L_max + 1, n_q)
                bessel_buffer = Vector{Float64}(undef, L_max + 1)
                kernel_table = Matrix{Complex{T}}(undef, n_q, n_triples) # Stores K^∥ for each frame.
                first = Array{Complex{T}, 3}(undef, n_transitions, n_q, n_keys) # Stores rotated (now pair-frame) coefficients for the "first" molecule. "f1".
                accum = Array{Complex{T}, 3}(undef, n_transitions, n_q, n_keys) # Stores sum_ℓ f1 K^∥ for each q and (l', μ). "Σ f1 K^∥".
                second = Array{Complex{T}, 3}(undef, n_q, n_keys, n_transitions) # Stores rotated (now pair-frame) coefficients for the "second" molecule. "f2".
                # The same two buffers, shaped as matrices for the closing gemm. This does not allocate, just reads from the above.
                accum_matrix = reshape(accum, n_transitions, n_q * n_keys)
                second_matrix = reshape(second, n_q * n_keys, n_transitions)
                overlaps = Matrix{Complex{T}}(undef, n_transitions, n_transitions)

                for job_idx in chunk:n_chunks:length(jobs)
                    # Get the indices for the current job.
                    (image_i, image_j, cell_idx) = jobs[job_idx]
                    lambdas_i = lambdas_of_image[image_i]
                    lambdas_j = lambdas_of_image[image_j]

                    separation = basis.images[image_i].translation -
                                 basis.images[image_j].translation - cells.vectors[cell_idx]
                    distance = norm(separation)

                    # Compute the Bessel functions at each q.
                    @inbounds for q_idx in 1:n_q
                        fill_spherical_bessel_column!(bessel_table, bessel_buffer, q_idx,
                                                      q_grid_invA[q_idx] * distance, L_max)
                    end

                    # Build the full kernel by multiplying the Gaunt part by the Bessel functions.
                    @inbounds for t in 1:n_triples
                        span = kernel.offsets[t]:(kernel.offsets[t + 1] - 1)
                        for q_idx in 1:n_q
                            total = zero(Complex{T})
                            for n in span
                                total += kernel.member_weight[n] * bessel_table[kernel.member_L[n] + 1, q_idx]
                            end
                            kernel_table[q_idx, t] = radial_weights[q_idx] * total
                        end
                    end

                    # Rotate to the pair frame.
                    blocks = wigner_d_blocks(rotor_to_z(separation / distance), l_max)

                    source_i = first_coefficients[image_i]
                    source_j = second_coefficients[image_j]

                    @inbounds for l in 0:l_max
                        block = (l * l + 1):((l + 1) * (l + 1))
                        width = 2 * l + 1
                        transposed_D = transpose(blocks[l + 1])
                        mul!(reshape(view(first, :, :, block), n_transitions * n_q, width),
                             reshape(view(source_i, :, :, block), n_transitions * n_q, width), transposed_D)
                        # The adjoint rather than the transpose, since we're rotating a conjugated coefficient. 
                        conjugated_D = adjoint(blocks[l + 1])
                        for slot in 1:n_transitions
                            mul!(view(second, :, block, slot),
                                 view(source_j, :, block, slot), conjugated_D)
                        end
                    end

                    # Compute the sum_ℓ f1 K^∥.
                    fill!(accum, zero(Complex{T}))
                    @inbounds for t in 1:n_triples
                        (l, l_prime, mu) = kernel.triples[t]
                        key_first = l * l + (l + mu) + 1
                        key_second = l_prime * l_prime + (l_prime + mu) + 1
                        for q_idx in 1:n_q
                            weight = kernel_table[q_idx, t]
                            @simd for slot in 1:n_transitions
                                accum[slot, q_idx, key_second] +=
                                    weight * first[slot, q_idx, key_first]
                            end
                        end
                    end

                    # Now contract with the second molecule coefficients.
                    mul!(overlaps, accum_matrix, second_matrix)

                    # Fill the coupling matrix.
                    @inbounds for b in 1:n_transitions, a in 1:n_transitions
                        lambda = lambdas_i[a]
                        lambda_prime = lambdas_j[b]
                        lambda <= lambda_prime || continue
                        value = prefactor * overlaps[a, b]
                        couplings[cell_idx, lambda, lambda_prime] = value
                        # The Hermitian partner lives in the opposite cell.
                        couplings[cells.negative_of[cell_idx], lambda_prime, lambda] = conj(value)
                    end
                end
            end
        end
    # Reset BLAS thread count.
    finally
        BLAS.set_num_threads(blas_threads)
    end

    return CrystalCouplings{T}(couplings, cells.vectors)
end

function subtract_self_term!(
        couplings::CrystalCouplings{T},
        cells::NeighbourCells{T},
        self_term::Matrix{Complex{T}},
    )::CrystalCouplings{T} where {T<:AbstractFloat}
    """
    Remove the self-interaction that the Ewald reciprocal sum includes.
    This is the "-δ_{A_i,B_j} J^LR_{A_i s, A_i t}(0)" part.

    # Arguments:
    - couplings::CrystalCouplings{T}: The short-range couplings, modified in place.
    - cells::NeighbourCells{T}: The lattice vectors, used to locate ΔR = 0.
    - self_term::Matrix{Complex{T}}: The self-interaction J^LR_{λλ'}(0) in eV.

    # Returns:
    - CrystalCouplings{T}: The same couplings, with the self term removed.
    """

    zero_cell = findfirst(index -> all(iszero, index), cells.indices)
    zero_cell === nothing && error("The neighbour list does not contain ΔR = 0, which should be impossible.")

    @views couplings.values[zero_cell, :, :] .-= self_term

    return couplings
end

end
