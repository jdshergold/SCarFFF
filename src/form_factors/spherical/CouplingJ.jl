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
using ..ConstructFSquaredTensor: load_gaunt_array
using ..DensityFormFactors: Delta_row
using ..ReadBasisSet: build_transition_pairs
using ..ProjectFSquared: quadrature_weights
using ..WignerRotations: WignerDBuffers, wigner_d_blocks!
using ...FastPowers: fast_i_pow

export NeighbourCells, enumerate_neighbour_cells, compute_couplings, compute_crystal_corrections,
       subtract_self_term!,
       image_translation_span, rotation_to_pair_frame

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
        span = max(span, norm(image_i.tau - image_j.tau))
    end

    return span
end

struct NeighbourCells{T<:AbstractFloat}
    """
    The lattice vectors ΔR included in the coupling sum, and the bookkeeping needed to relate a cell
    to its negative, ΔR → -ΔR.

    # Fields:
    - Delta_R::Vector{SVector{3, T}}: The lattice vectors ΔR, in Å.
    - indices::Vector{NTuple{3, Int}}: The integer coordinates of each ΔR in the lattice basis.
    - negative_of::Vector{Int}: For each cell, the index of the cell holding -ΔR.
    """
    Delta_R::Vector{SVector{3, T}}
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
        lattice.V_cell / norm(cross(a2, a3)),
        lattice.V_cell / norm(cross(a3, a1)),
        lattice.V_cell / norm(cross(a1, a2)),
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

function rotation_to_pair_frame(direction::SVector{3, T})::Quaternionic.Rotor{T} where {T<:AbstractFloat}
    """
    The pair-frame rotation S⁻¹ mapping a unit separation onto ẑ,

        S⁻¹ d̂ = ẑ.

    A short note on quaternions. A quaternion (w, x, y, z) in our case represents
    a rotation about the axis (x, y, z), normalised, by an angle θ = 2 arccos(w).

    # Arguments:
    - direction::SVector{3, T}: A unit vector.

    # Returns:
    - Quaternionic.Rotor{T}: The rotor representing S⁻¹.
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

struct PairFrameP{T<:AbstractFloat}
    """
    The q-independent Gaunt weights in the pair-frame plane-wave factor P̄.
    Each entry is i^L √((2L+1)/4π) 𝒢_{ℓμ,L0}^{ℓ′μ}.
    Multiplying by j_L(qd) and summing over L gives P̄_{ℓℓ′,μ}(q).

    # Fields:
    - triples::Vector{NTuple{3, Int}}: The surviving (l, l', μ), sorted by destination key.
    - offsets::Vector{Int}: Start of each triple's entries in the two member arrays, plus a final stop.
    - member_L::Vector{Int}: The Bessel order L of each entry.
    - member_weight::Vector{Complex{T}}: i^L √((2L+1)/4π) 𝒢_{ℓμ,L0}^{ℓ′μ}.
    """
    triples::Vector{NTuple{3, Int}}
    offsets::Vector{Int}
    member_L::Vector{Int}
    member_weight::Vector{Complex{T}}
end

function build_pair_frame_P(gaunt_path::String, ::Type{T})::PairFrameP{T} where {T<:AbstractFloat}
    """
    Group the M = 0 Gaunt coefficients into the pair-frame P_pair table.
    The Gaunt angular cutoffs are (l_max, 2 * l_max, l_max) for (ℓ,L,ℓ′).

    # Arguments:
    - gaunt_path::String: Path to the coupling-shape Gaunt coefficients.
    - T::Type: The float type.

    # Returns:
    - PairFrameP{T}: The reduced table.
    """

    gaunt = load_gaunt_array(gaunt_path, T)

    grouped = Dict{NTuple{3, Int}, Vector{Int}}()
    # Read 𝒢_{ℓμ,L0}^{ℓ′μ}, with (l1,l2,l3) = (ℓ,L,ℓ′).
    # Its m1 and m3 are μ, and the inferred m2 is the plane-wave M = 0.
    for idx in eachindex(gaunt.Gaunt_values)
        M = gaunt.m3[idx] - gaunt.m1[idx]
        iszero(M) || continue
        l = gaunt.l1[idx]
        l_prime = gaunt.l3[idx]
        mu = gaunt.m1[idx]
        key = (l, l_prime, mu)
        # This is some shorthand. It means, if grouped[key] doesn't exist, create it as an empty Int vector, then push idx onto it.
        # If it does exist, just push idx onto it. Saves some lookups.
        push!(get!(grouped, key, Int[]), idx)
    end

    # Define the destination key function. Maps (l', μ) to l'^2 + (l' + μ) + 1.
    destination_key(triple) = triple[2] * triple[2] + (triple[2] + triple[3]) + 1
    # Sort the keys by μ first, then by the destination key. Speeds up memory access later.
    triples = sort!(collect(keys(grouped)); by = triple -> (triple[3], destination_key(triple)))

    # Now flatten the grouped data into the P_pair table.
    offsets = Vector{Int}(undef, length(triples) + 1)
    member_L = Int[]
    member_weight = Complex{T}[]
    for (position, triple) in enumerate(triples)
        offsets[position] = length(member_L) + 1
        for idx in grouped[triple]
            L = gaunt.l2[idx]
            push!(member_L, L)
            push!(member_weight,
                  fast_i_pow(L, T) * sqrt((2 * L + 1) / (4 * T(π))) * gaunt.Gaunt_values[idx])
        end
    end
    offsets[end] = length(member_L) + 1

    return PairFrameP{T}(triples, offsets, member_L, member_weight)
end

function radial_quadrature(
        q_grid_invA::Vector{T},
        parameters::Union{Nothing, EwaldParameters{T}},
    )::Vector{T} where {T<:AbstractFloat}
    """
    The weights of one coupling's q integral, including its short-range Ewald kernel.

    # Arguments:
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å.
    - parameters::Union{Nothing, EwaldParameters{T}}: The Ewald split, or nothing for the bare
      Coulomb sum, which uses Simpson's rule instead.

    # Returns:
    - Vector{T}: The weights, in inverse Å.
    """

    parameters === nothing && return quadrature_weights(q_grid_invA, false)

    # Use Δq at interior nodes and Δq/2 at the endpoints, as assumed by the SR spacing estimate.
    weights = fill(q_grid_invA[2] - q_grid_invA[1], length(q_grid_invA))
    weights[1] /= 2
    weights[end] /= 2

    return weights .* short_range_kernel(q_grid_invA, parameters.eta)
end

function fill_pair_P!(
        P_table::Matrix{Complex{T}},
        bessel_table::Matrix{T},
        P_pair::PairFrameP{T},
    ) where {T<:AbstractFloat}
    """
    Multiply the Gaunt part of P̄ by its Bessel functions.

    Nothing here depends on η, so one table serves both couplings however their splits differ. Each
    coupling's quadrature weights and short-range kernel are carried by the form factor it
    contracts this against instead.

    # Arguments:
    - P_table::Matrix{Complex{T}}: Preallocated P̄(q) table to overwrite, dimensionless.
    - bessel_table::Matrix{T}: j_L(qd), indexed by L and q.
    - P_pair::PairFrameP{T}: The grouped Gaunt coefficients for each (l, l_prime, mu).

    # Returns:
    - Nothing. P_table is overwritten.
    """

    n_q = size(P_table, 1)
    @inbounds for triple in eachindex(P_pair.triples)
        # The triple is (l, l_prime, mu). Get the Gaunt coefficient indices for this set.
        member_start = P_pair.offsets[triple]
        member_stop = P_pair.offsets[triple + 1] - 1
        for q_idx in 1:n_q
            total = zero(Complex{T})
            for member in member_start:member_stop
                total += P_pair.member_weight[member] *
                         bessel_table[P_pair.member_L[member] + 1, q_idx]
            end
            P_table[q_idx, triple] = total
        end
    end

    return nothing
end

function add_D_potential!(
        potential::Array{Complex{T}, 3},
        image_idx::Int,
        Xi_source::Matrix{Complex{T}},
        blocks::Vector{Matrix{Complex{T}}},
        P_pair::PairFrameP{T},
        P_table::Matrix{Complex{T}},
        Xi_pair::Matrix{Complex{T}},
        Xi_potential::Matrix{Complex{T}},
        l_max::Int,
    ) where {T<:AbstractFloat}
    """
    Add the Coulomb potential of one neighbouring molecule's charge density, P̄ Ξ*, in the crystal
    frame.

    Ξ holds the ground-state charge density of the neighbour and P̄ is the Coulomb kernel, so P̄ Ξ*
    is the electrostatic potential that density sets up across the molecule we are working on. Its
    transition density Δ then sits in that potential, which is what D measures.

    P̄ is applied once to Ξ rather than once to every Δ_{st}, as the potential does not depend on
    the pair of states. It is then rotated back into the crystal frame.

        V_a(q)  = Σ_b P̄_ab(q) Ξ_b*(q)                 in the pair frame,
        W_m(q)  = Σ_μ D^(l)_μm V_μ(q)                 back in the crystal frame,
        D_{st}  = Σ_m Δ_{st,m}(q) W_m(q).

    Rotating the one potential backwards rather than every Δ forwards is the same arithmetic, and it saves holding
    a pair-frame copy of Δ for every pair of states.

    # Arguments:
    - potential::Array{Complex{T},3}: Task-local W, with dimensions (n_q, n_keys, n_images).
    - image_idx::Int: The molecule the potential is evaluated at.
    - Xi_source::Matrix{Complex{T}}: Conjugated Ξ coefficients of the neighbouring image, already
      carrying D's quadrature weights and short-range kernel.
    - blocks::Vector{Matrix{Complex{T}}}: Wigner matrices for this pair frame.
    - P_pair::PairFrameP{T}: The grouped Gaunt part of P̄.
    - P_table::Matrix{Complex{T}}: P̄(q) for this separation.
    - Xi_pair::Matrix{Complex{T}}: Buffer for pair-frame Ξ*.
    - Xi_potential::Matrix{Complex{T}}: Buffer for P̄ Ξ* in the pair frame.
    - l_max::Int: The largest angular mode.

    # Returns:
    - Nothing. potential is accumulated in place.
    """

    n_q = size(Xi_source, 1)

    # Ξ is stored conjugated, so its coefficient rotation uses D*.
    @inbounds for l in 0:l_max
        block = (l * l + 1):((l + 1) * (l + 1))
        D = blocks[l + 1]
        mul!(view(Xi_pair, :, block), view(Xi_source, :, block), adjoint(D))
    end

    # Applying P̄ to the single Xi object avoids repeating the expensive Gaunt contraction for
    # every pair of states.
    fill!(Xi_potential, zero(Complex{T}))
    @inbounds for triple in eachindex(P_pair.triples)
        (l, l_prime, mu) = P_pair.triples[triple]
        key_first = l * l + (l + mu) + 1
        key_second = l_prime * l_prime + (l_prime + mu) + 1
        for q_idx in 1:n_q
            Xi_potential[q_idx, key_first] +=
                P_table[q_idx, triple] * Xi_pair[q_idx, key_second]
        end
    end

    # Rotate the potential back out of the pair frame and add it to the total this molecule sits in,
    # so Δ can stay where it is and be contracted once at the end. The image sits last so that a
    # single molecule's slice is one contiguous matrix, which the multiply can accumulate into.
    @inbounds for l in 0:l_max
        block = (l * l + 1):((l + 1) * (l + 1))
        D = blocks[l + 1]
        mul!(view(potential, :, block, image_idx), view(Xi_potential, :, block), D,
             one(Complex{T}), one(Complex{T}))
    end

    return nothing
end

function contract_D_potential!(
        D_blocks::Array{Complex{T}, 3},
        potential::Array{Complex{T}, 3},
        rotated_Delta::Array{Complex{T}, 3},
        transitions_of_pair::Vector{NTuple{2, Int}},
        overlap::Vector{Complex{T}},
        prefactor::Complex{T},
    ) where {T<:AbstractFloat}
    """
    Contract the total potential each molecule sits in with its Δ, and place the result in D.

    add_D_potential! has already added the contributions of every neighbour, and Δ is the same
    whichever neighbour produced the potential, so the whole sum over neighbours needs this
    contraction only once per molecule.

    # Arguments:
    - D_blocks::Array{Complex{T},3}: D, with dimensions (n_images, n_transitions, n_transitions).
    - potential::Array{Complex{T},3}: The summed W, with dimensions (n_q, n_keys, n_images).
    - rotated_Delta::Array{Complex{T},3}: Δ for every image, one row per (image, pair).
    - transitions_of_pair::Vector{NTuple{2,Int}}: The (s, t) carried by each pair index.
    - overlap::Vector{Complex{T}}: Buffer for one overlap per pair of states.
    - prefactor::Complex{T}: The Coulomb prefactor in eV Angstrom.

    # Returns:
    - Nothing. D_blocks is filled in place.
    """

    n_keys, n_images = size(potential, 2), size(potential, 3)
    n_pairs = length(transitions_of_pair)

    for image_idx in 1:n_images
        # The rows of one image sit together, so Δ can be read straight out of the rotated tensor
        # one spherical harmonic at a time, with no copy.
        rows = ((image_idx - 1) * n_pairs + 1):(image_idx * n_pairs)
        fill!(overlap, zero(Complex{T}))
        for key in 1:n_keys
            mul!(overlap, view(rotated_Delta, rows, :, key),
                 view(potential, :, key, image_idx), one(Complex{T}), one(Complex{T}))
        end

        @inbounds for pair in 1:n_pairs
            (s, t) = transitions_of_pair[pair]
            value = prefactor * overlap[pair]
            D_blocks[image_idx, s, t] += value
            s == t && continue
            D_blocks[image_idx, t, s] += conj(value)
        end
    end

    return nothing
end

function _compute_crystal_corrections(
        rotated_R::Array{Complex{T}, 3},
        basis::CrystalExcitationBasis{T},
        cells::NeighbourCells{T},
        q_grid_invA::Vector{T},
        l_max::Int,
        gaunt_path::String;
        J_parameters::Union{Nothing, EwaldParameters{T}} = nothing,
        D_parameters::Union{Nothing, EwaldParameters{T}} = nothing,
        rotated_Delta::Union{Nothing, Array{Complex{T}, 3}} = nothing,
        rotated_Xi::Union{Nothing, Array{Complex{T}, 3}} = nothing,
    ) where {T<:AbstractFloat}
    """
    Compute the short-range Frenkel couplings and, when supplied, the same-molecule D blocks in each pair frame.

        J_SR_{A_i s, B_j t}(ΔR) = (2 α_EM / π) Σ_{ℓ,ℓ'} Σ_μ ∫dq [1 - exp(-q²/4η²)]
                                      f̄^{(A_i,s)}_{ℓμ,αβ}(q) P̄^{αβ}_{ℓℓ',μ}(q) f̄^{(B_j,t)*}_{ℓ'μ,αβ}(q),

    where d_{αβ} = τ_λ_i - τ_λ_j′ - ΔR is the vector between the two molecules, and

        P̄^{αβ}_{ℓℓ',μ}(q) = Σ_L i^L √((2L+1)/4π) j_L(q d_{αβ}) G_{L0,ℓμ}^{ℓ'μ},

        f̄^{(A_i,s)}_{ℓμ,αβ}(q)  = Σ_ν D_{μν}(S^{-1}_{αβ}) f^{(A_i,s)}_{ℓν}(q).

    The short-range Ewald factor is 1 - exp(-q²/4η²), with η the Ewald parameter. J and D each carry
    their own split, to not waste work on one of the terms. Usually D is cheaper.

    To ensure that the SR integral converges to the require precision, its q grid can differ
    from that of the structure function.

    # Arguments:
    - rotated_R::Array{Complex{T}, 3}: The image coefficients f, with dimensions (n_lambda_i, n_q, n_keys).
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - cells::NeighbourCells{T}: The lattice vectors to sum over.
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å.
    - l_max::Int: The maximum angular momentum mode of the form factor.
    - gaunt_path::String: Path to the coupling-shape Gaunt coefficients.
    - J_parameters::Union{Nothing, EwaldParameters{T}}: J's Ewald split, or nothing for the bare
      Coulomb P_pair and hence the full, only conditionally convergent, lattice sum.
    - D_parameters::Union{Nothing, EwaldParameters{T}}: D's Ewald split, or nothing for the bare sum.
    - rotated_Delta: The rotated Δ coefficients, or nothing for a J-only calculation.
    - rotated_Xi: The rotated neutral ground-charge coefficients, or nothing for a J-only calculation.

    # Returns:
    - Tuple: The couplings and the short-range D blocks, both in eV.
    """

    # Get the dimensions.
    n_lambda_i = length(basis.Delta_E_monomer)
    n_q = length(q_grid_invA)
    n_neighbour_cells = length(cells.Delta_R)
    n_keys = (l_max + 1)^2
    L_max = 2 * l_max
    have_Delta_and_Xi = rotated_Delta !== nothing || rotated_Xi !== nothing
    (rotated_Delta === nothing) == (rotated_Xi === nothing) || error(
        "Both the Δ and Ξ tensors are required to compute D.")
    # The two couplings share this pass, so an Ewald split for one without the other would leave
    # the second as a truncated bare Coulomb sum, which does not converge.
    !have_Delta_and_Xi || (J_parameters === nothing) == (D_parameters === nothing) || error(
        "J and D must either both carry an Ewald split, or neither.")

    # Check that the rotated coefficients have the right dimensions.
    (l_max + 1)^2 == size(rotated_R, 3) ||
        error("The rotated coefficients have $(size(rotated_R, 3)) keys, which does not match l_max = $(l_max).")

    # Build the pair-frame P_pair, which is P̄ up to the Bessel function.
    P_pair = build_pair_frame_P(gaunt_path, T)
    n_triples = length(P_pair.triples)

    # The q integral of each coupling, weights and short-range kernel together. Each is folded into
    # the form factor on the right of its own integral (either Ξ or f) further down, so that P̄ stays shared.
    radial_weights_J = radial_quadrature(q_grid_invA, J_parameters)
    radial_weights_D = radial_quadrature(q_grid_invA, D_parameters)
    prefactor = Complex{T}(2 * ALPHA_EM / π * HBAR_C_EV_ANGSTROM)

    n_images = length(basis.images)
    # Find the lambdas for each image.
    lambda_i_of_image = [findall(==(image_idx), basis.image_of) for image_idx in 1:n_images]
    # Every image carries the same transitions, which is what lets the buffers below be fixed size.
    n_transitions = basis.n_transitions
    all(==(n_transitions), length.(lambda_i_of_image)) || error(
        "Each image should carry $(n_transitions) transitions, but the counts are " *
        "$(length.(lambda_i_of_image))."
    )

    # Δ carries one row per (s, t) pair of each image, rather than one per λ_i.
    pairs = build_transition_pairs(n_transitions)
    n_pairs = length(pairs.transitions_of_pair)
    if have_Delta_and_Xi
        size(rotated_Delta) == (n_images * n_pairs, n_q, n_keys) || error(
            "The rotated Δ tensor has size $(size(rotated_Delta)), expected $((n_images * n_pairs, n_q, n_keys)).")
        size(rotated_Xi) == (n_images, n_q, n_keys) || error(
            "The rotated Ξ tensor has size $(size(rotated_Xi)), expected $((n_images, n_q, n_keys)).")
    end

    # Store the rotated f_sq_lm (R tensor) coefficients for all molecules in the unit cell.
    # The first are the unconjugated, second are conjugated. We store them with different indices to speed up
    # matmuls later, by saving on transposing and conjugating. The second copy also carries J's radial weights.
    first_coefficients = [Array{Complex{T}, 3}(undef, length(lambda_i_of_image[i]), n_q, n_keys)
                          for i in 1:n_images]
    second_coefficients = [Array{Complex{T}, 3}(undef, n_q, n_keys, length(lambda_i_of_image[i]))
                           for i in 1:n_images]
    for image_idx in 1:n_images, (slot, lambda_i) in enumerate(lambda_i_of_image[image_idx])
        @inbounds for key in 1:n_keys, q_idx in 1:n_q
            value = rotated_R[lambda_i, q_idx, key]
            first_coefficients[image_idx][slot, q_idx, key] = value
            second_coefficients[image_idx][q_idx, key, slot] = radial_weights_J[q_idx] * conj(value)
        end
    end

    # Ξ is conjugated once here, since it always appears that way, and carries D's radial weights
    # for the same reason the second f copy carries J's. Δ needs no copy at all: it stays
    # in the crystal frame, where the potential is rotated back to meet it.
    Xi_coefficients = have_Delta_and_Xi ?
        [Matrix{Complex{T}}(undef, n_q, n_keys) for _ in 1:n_images] : nothing
    if have_Delta_and_Xi
        for image_idx in 1:n_images
            @inbounds for key in 1:n_keys, q_idx in 1:n_q
                Xi_coefficients[image_idx][q_idx, key] =
                    radial_weights_D[q_idx] * conj(rotated_Xi[image_idx, q_idx, key])
            end
        end
    end

    R_max_J = J_parameters === nothing ? T(Inf) : J_parameters.R_max
    R_max_D = zero(T)
    if have_Delta_and_Xi
        R_max_D = D_parameters === nothing ? T(Inf) : D_parameters.R_max
    end

    # One job per (image pair, cell). Molecules do not interact with themselves, and under
    # the Ewald split the cutoff applies to the molecular separation, not to the lattice vector.
    jobs = NTuple{3, Int}[]
    for image_i in 1:n_images, image_j in image_i:n_images, cell_idx in 1:n_neighbour_cells
        # Skip i = j and zero lattice vectors.
        image_i == image_j && all(iszero, cells.indices[cell_idx]) && continue
        # Build τ_i - τ_j - ΔR.
        separation = basis.images[image_i].tau - basis.images[image_j].tau -
                     cells.Delta_R[cell_idx]
        distance = norm(separation)
        distance > eps(T) || error(
            "Molecules $(image_i) and $(image_j) coincide at ΔR = $(cells.indices[cell_idx]), " *
            "which should be impossible. Check the translation vectors in the crystal metadata."
        )
        # Check that the molecule pair is within the cutoff of at least one coupling.
        distance > max(R_max_J, R_max_D) && continue
        push!(jobs, (image_i, image_j, cell_idx))
    end

    # Only do work for cells within J's cutoff.
    keep_cell = falses(n_neighbour_cells)
    for (image_i, image_j, cell_idx) in jobs
        separation = basis.images[image_i].tau - basis.images[image_j].tau -
                     cells.Delta_R[cell_idx]
        norm(separation) <= R_max_J || continue
        keep_cell[cell_idx] = true
        keep_cell[cells.negative_of[cell_idx]] = true
    end
    # ΔR = 0 is always kept.
    zero_cell = findfirst(index -> all(iszero, index), cells.indices)
    zero_cell === nothing &&
        error("The neighbour list does not contain ΔR = 0, which should be impossible.")
    keep_cell[zero_cell] = true
    kept_cells = findall(keep_cell)
    # Build the map from each kept cell to its slot in the full cell list.
    slot_of_cell = zeros(Int, n_neighbour_cells)
    for (slot, cell_idx) in enumerate(kept_cells)
        slot_of_cell[cell_idx] = slot
    end

    couplings = zeros(Complex{T}, length(kept_cells), n_lambda_i, n_lambda_i)
    n_chunks = min(nthreads(), max(length(jobs), 1))
    potential_chunks = have_Delta_and_Xi ?
        [zeros(Complex{T}, n_q, n_keys, n_images) for _ in 1:n_chunks] : nothing

    # Avoid BLAS oversubscription.
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        @sync for chunk in 1:n_chunks
            Threads.@spawn begin
                # Allocate buffers for each thread.
                bessel_table = Array{T, 2}(undef, L_max + 1, n_q)
                bessel_buffer = Vector{Float64}(undef, L_max + 1)
                P_table = Matrix{Complex{T}}(undef, n_q, n_triples) # Stores P̄ for each frame.
                wigner_buffers = WignerDBuffers(T, l_max)
                first = Array{Complex{T}, 3}(undef, n_transitions, n_q, n_keys) # Stores rotated (now pair-frame) coefficients for the "first" molecule. "f1".
                accum = Array{Complex{T}, 3}(undef, n_transitions, n_q, n_keys) # Stores sum_ℓ f1 P̄ for each q and (l', μ). "Σ f1 P̄".
                second = Array{Complex{T}, 3}(undef, n_q, n_keys, n_transitions) # Stores rotated (now pair-frame) coefficients for the "second" molecule. "f2".
                # The same two buffers, shaped as matrices for the closing gemm. This does not allocate, just reads from the above.
                accum_matrix = reshape(accum, n_transitions, n_q * n_keys)
                second_matrix = reshape(second, n_q * n_keys, n_transitions)
                overlaps = Matrix{Complex{T}}(undef, n_transitions, n_transitions)

                # D has one Δ per pair of states on the left and one Ξ on the right. Only the Ξ side
                # needs pair-frame buffers, since the potential is rotated back to meet Δ.
                Xi_pair = have_Delta_and_Xi ? Matrix{Complex{T}}(undef, n_q, n_keys) : nothing
                Xi_potential = have_Delta_and_Xi ? similar(Xi_pair) : nothing
                local_potential = have_Delta_and_Xi ? potential_chunks[chunk] : nothing

                for job_idx in chunk:n_chunks:length(jobs)
                    # Get the indices for the current job.
                    (image_i, image_j, cell_idx) = jobs[job_idx]
                    lambda_i_on_first = lambda_i_of_image[image_i]
                    lambda_i_on_second = lambda_i_of_image[image_j]

                    separation = basis.images[image_i].tau -
                                 basis.images[image_j].tau - cells.Delta_R[cell_idx]
                    distance = norm(separation)
                    # Each coupling keeps only the pairs inside its own reach.
                    inside_J = distance <= R_max_J
                    inside_D = have_Delta_and_Xi && distance <= R_max_D

                    # Compute the Bessel functions at each q.
                    @inbounds for q_idx in 1:n_q
                        fill_spherical_bessel_column!(bessel_table, bessel_buffer, q_idx,
                                                      q_grid_invA[q_idx] * distance, L_max)
                    end

                    # Build the full P_pair by multiplying the Gaunt part by the Bessel functions.
                    fill_pair_P!(P_table, bessel_table, P_pair)

                    # Rotate to the pair frame.
                    S_inverse = rotation_to_pair_frame(separation / distance)
                    blocks = wigner_d_blocks!(wigner_buffers, S_inverse)

                    if inside_J
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

                        # Compute the sum_ℓ f1 P̄.
                        fill!(accum, zero(Complex{T}))
                        @inbounds for t in 1:n_triples
                            (l, l_prime, mu) = P_pair.triples[t]
                            key_first = l * l + (l + mu) + 1
                            key_second = l_prime * l_prime + (l_prime + mu) + 1
                            for q_idx in 1:n_q
                                weight = P_table[q_idx, t]
                                @simd for slot in 1:n_transitions
                                    accum[slot, q_idx, key_second] +=
                                        weight * first[slot, q_idx, key_first]
                                end
                            end
                        end

                        # Now contract with the second molecule coefficients.
                        mul!(overlaps, accum_matrix, second_matrix)

                        # Fill the coupling matrix, in the trimmed cell list's indexing.
                        slot = slot_of_cell[cell_idx]
                        negative_slot = slot_of_cell[cells.negative_of[cell_idx]]
                        @inbounds for b in 1:n_transitions, a in 1:n_transitions
                            lambda_i = lambda_i_on_first[a]
                            lambda_j = lambda_i_on_second[b]
                            lambda_i <= lambda_j || continue
                            value = prefactor * overlaps[a, b]
                            couplings[slot, lambda_i, lambda_j] = value
                            # The Hermitian partner lives in the opposite cell.
                            couplings[negative_slot, lambda_j, lambda_i] = conj(value)
                        end
                    end

                    if inside_D
                        # The potential of molecule j's charge density across molecule i, in the
                        # pair frame already built for J_ij.
                        add_D_potential!(
                            local_potential, image_i,
                            Xi_coefficients[image_j], blocks, P_pair, P_table,
                            Xi_pair, Xi_potential, l_max)

                        # The same the other way round, molecule i across molecule j, at separation
                        # -d. The pair frame is different, so it needs its own rotation.
                        if image_i != image_j
                            reverse_S_inverse = rotation_to_pair_frame(-separation / distance)
                            reverse_blocks = wigner_d_blocks!(wigner_buffers, reverse_S_inverse)
                            add_D_potential!(
                                local_potential, image_j,
                                Xi_coefficients[image_i], reverse_blocks, P_pair, P_table,
                                Xi_pair, Xi_potential, l_max)
                        end
                    end
                end
            end
        end
    # Reset BLAS thread count.
    finally
        BLAS.set_num_threads(blas_threads)
    end

    # Sum the per-task potentials, then contract each image's total with its Δ just once.
    # Δ P̄ Ξ* comes out real, so check that and then force it.
    D_blocks = zeros(T, n_images, n_transitions, n_transitions)
    if have_Delta_and_Xi
        summed_potential = potential_chunks[1]
        for chunk in 2:n_chunks
            summed_potential .+= potential_chunks[chunk]
        end
        total = zeros(Complex{T}, n_images, n_transitions, n_transitions)
        contract_D_potential!(total, summed_potential, rotated_Delta, pairs.transitions_of_pair,
                              Vector{Complex{T}}(undef, n_pairs), prefactor)
        scale = max(maximum(abs, real(total)), one(T))
        imaginary_residual = maximum(abs, imag(total))
        tolerance = T(1000) * eps(T) * scale
        imaginary_residual <= tolerance || error(
            "The short-range D has imaginary residual $(imaginary_residual), tolerance $(tolerance).")
        D_blocks .= real(total)
    end

    return CrystalCouplings{T}(couplings, cells.Delta_R[kept_cells]), D_blocks
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
    Compute the direct or Ewald short-range transition couplings J.

    This is the J-only interface. It uses the same pair implementation as the combined
    crystal correction without constructing or returning D.

    # Arguments:
    - rotated_R::Array{Complex{T},3}: Rotated transition coefficients for every lambda_i.
    - basis::CrystalExcitationBasis{T}: The local excitation basis.
    - cells::NeighbourCells{T}: Lattice cells included in the real-space sum.
    - q_grid_invA::Vector{T}: The q grid, in inverse Angstroms.
    - l_max::Int: The largest angular mode.
    - gaunt_path::String: Path to the coupling Gaunt coefficients.
    - parameters::Union{Nothing,EwaldParameters{T}}: The Ewald split, or nothing for direct J.

    # Returns:
    - CrystalCouplings{T}: J for every cell and pair of local excitations, in eV.
    """

    couplings, _ = _compute_crystal_corrections(
        rotated_R, basis, cells, q_grid_invA, l_max, gaunt_path; J_parameters = parameters)
    return couplings
end

function compute_crystal_corrections(
        rotated_R::Array{Complex{T}, 3},
        rotated_Delta::Array{Complex{T}, 3},
        rotated_Xi::Array{Complex{T}, 3},
        basis::CrystalExcitationBasis{T},
        cells::NeighbourCells{T},
        q_grid_invA::Vector{T},
        l_max::Int,
        gaunt_path::String;
        J_parameters::Union{Nothing, EwaldParameters{T}} = nothing,
        D_parameters::Union{Nothing, EwaldParameters{T}} = nothing,
    ) where {T<:AbstractFloat}
    """
    Compute J and the short-range D in one pair pass.

        J: R P_pair R*,             D_SR: Delta P_pair Xi*.

    The separation, Bessel table, pair-frame P table and pair-frame rotations are shared, even where
    the two couplings are split at different η, since none of them depends on η. For D, P̄ is
    applied once to Ξ and the resulting potential is dotted with every Δ.

    # Arguments:
    - rotated_R::Array{Complex{T},3}: Rotated transition coefficients for every lambda_i.
    - rotated_Delta::Array{Complex{T},3}: Rotated Δ, one row per image and pair of states.
    - rotated_Xi::Array{Complex{T},3}: Rotated Ξ for every molecular image.
    - basis::CrystalExcitationBasis{T}: The local excitation basis.
    - cells::NeighbourCells{T}: Lattice cells included in the real-space sum.
    - q_grid_invA::Vector{T}: The q grid, in inverse Angstroms.
    - l_max::Int: The largest angular mode.
    - gaunt_path::String: Path to the coupling Gaunt coefficients.
    - J_parameters::Union{Nothing,EwaldParameters{T}}: J's Ewald split, or nothing for direct sums.
    - D_parameters::Union{Nothing,EwaldParameters{T}}: D's Ewald split, or nothing for direct sums.

    # Returns:
    - Tuple: CrystalCouplings J and the short-range D blocks, with dimensions
      (n_images, n_transitions, n_transitions), both in eV.
    """

    return _compute_crystal_corrections(
        rotated_R, basis, cells, q_grid_invA, l_max, gaunt_path;
        J_parameters = J_parameters, D_parameters = D_parameters,
        rotated_Delta = rotated_Delta, rotated_Xi = rotated_Xi)
end

function subtract_self_term!(
        couplings::CrystalCouplings{T},
        J_self::Matrix{Complex{T}},
    )::CrystalCouplings{T} where {T<:AbstractFloat}
    """
    Remove the self-interaction that the Ewald reciprocal sum includes.
    This is the "-δ_{A_i,B_j} J^self_{A_i s, A_i t}" part.

    # Arguments:
    - couplings::CrystalCouplings{T}: The short-range couplings, modified in place.
    - J_self::Matrix{Complex{T}}: The self-interaction J^self_{λ_iλ_j′} in eV.

    # Returns:
    - CrystalCouplings{T}: The same couplings, with the self term removed.
    """

    zero_cell = findfirst(Delta_R -> iszero(Delta_R), couplings.Delta_R)
    zero_cell === nothing && error("The couplings do not contain ΔR = 0, which should be impossible.")

    @views couplings.J_deltaR[zero_cell, :, :] .-= J_self

    return couplings
end

end
