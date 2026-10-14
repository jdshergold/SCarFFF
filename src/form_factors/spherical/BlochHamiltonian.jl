# This module builds and diagonalises the Frenkel exciton Bloch Hamiltonian for a crystal.

module BlochHamiltonian

using StaticArrays
using Quaternionic
using LinearAlgebra
using LinearAlgebra: dot, mul!
using FastLapackInterface: HermitianEigenWs

using ..Ewald: EwaldLongRangeData, EwaldLongRangeBuffers, add_long_range!

export CrystalImage, CrystalExcitationBasis, build_excitation_basis, CrystalCouplings,
       set_D_couplings!, build_bloch_hamiltonian!, solve_bloch_hamiltonian!,
       solve_bloch_energies!, BlochEigensystem

struct CrystalCouplings{T<:AbstractFloat}
    """
    The intermolecular couplings J_{λλ'}(ΔR), and the lattice vectors they are indexed by.

    # Fields:
    - values::Array{Complex{T}, 3}: The couplings in eV, with dimensions (n_cells, n_lambda, n_lambda).
    - cell_vectors::Vector{SVector{3, T}}: The lattice vector ΔR for each cell, in Å.
    """
    values::Array{Complex{T}, 3}
    cell_vectors::Vector{SVector{3, T}}
end

struct CrystalImage{T<:AbstractFloat}
    """
    One molecule in the unit cell: the (A, i) label of the theory.

    # Fields:
    - conformer_index::Int: Which reference conformer (A) this is an image of.
    - rotation::Quaternionic.Rotor{T}: The proper rotation R̃_{A,i} mapping the reference monomer onto this image.
    - det_rotation::T: det(R_{A,i}), which is -1 for improper operations.
    - translation::SVector{3, T}: The translation τ_{A,i} from the cell origin to this image, in Å.
    """
    conformer_index::Int
    rotation::Quaternionic.Rotor{T}
    det_rotation::T
    translation::SVector{3, T}
end

struct CrystalExcitationBasis{T<:AbstractFloat}
    """
    The localised excitation basis λ = (A, i, s), flattened to a single index.

    The ordering defined here is used by both the Bloch Hamiltonian and the
    coherent form factor sum, so that the eigenvector components line up with the molecular
    amplitudes. It is image-major, so λ runs over transitions fastest, then over images.

    For example, image_of[λ] gives the image index (A, i) for each λ, and transition_of[λ] gives the transition index s for each λ.
    In a unit cell with say, 2 molecules, and 3 transitions, this would have 6 entries.
    They would be ordered as follows:

    λ = 1-3: image 1, transition 1-3,
    λ = 4-6: image 2, transition 1-3.

    To get the actual information about a specific image, you would use something like images[image_of[λ]].

    # Fields:
    - image_of::Vector{Int}: The image index (A, i) for each λ.
    - transition_of::Vector{Int}: The transition index s for each λ, as a position in the requested transition list.
    - lambda_of::Matrix{Int}: The λ carrying each (image, transition), which inverts the two above.
    - energies::Vector{T}: The monomer excitation energy E_{A,s} for each λ, in eV.
    - D_couplings::Matrix{T}: The same-molecule couplings D_{λλ'} in eV, non-zero only for λ and λ'
      on the same image.
    - images::Vector{CrystalImage{T}}: The images themselves.
    - n_transitions::Int: The number of transitions per image.
    """
    image_of::Vector{Int}
    transition_of::Vector{Int}
    lambda_of::Matrix{Int}
    energies::Vector{T}
    D_couplings::Matrix{T}
    images::Vector{CrystalImage{T}}
    n_transitions::Int
end

function build_excitation_basis(
        images::Vector{CrystalImage{T}},
        conformer_transition_energies::Vector{Vector{T}},
    )::CrystalExcitationBasis{T} where {T<:AbstractFloat}
    """
    Flatten the localised excitations (A, i, s) into a single index λ.

    # Arguments:
    - images::Vector{CrystalImage{T}}: The molecules in the unit cell.
    - conformer_transition_energies::Vector{Vector{T}}: Excitation energies in eV, indexed [conformer][transition].

    # Returns:
    - CrystalExcitationBasis{T}: The flattened basis.
    """

    isempty(images) && error("Cannot build an excitation basis with no molecules in the unit cell.")

    n_transitions = length(conformer_transition_energies[1])
    # Check that all conformers have the same number of transitions.
    for (conformer_idx, energies) in enumerate(conformer_transition_energies)
        length(energies) == n_transitions ||
            error("Conformer $(conformer_idx) has $(length(energies)) transitions, but conformer 1 has $(n_transitions). All conformers must be computed for the same transitions.")
    end

    # Allocate the arrays for the flattened basis. 
    n_lambda = length(images) * n_transitions
    image_of = Vector{Int}(undef, n_lambda)
    transition_of = Vector{Int}(undef, n_lambda)
    lambda_of = Matrix{Int}(undef, length(images), n_transitions)
    energies = Vector{T}(undef, n_lambda)

    lambda = 0
    for (image_idx, image) in enumerate(images)
        conformer_idx = image.conformer_index
        # Check that the conformer index is valid and that there are transition energies for it.
        conformer_idx in eachindex(conformer_transition_energies) ||
            error("Image $(image_idx) refers to conformer $(conformer_idx), which has no transition energies.")

        for transition_idx in 1:n_transitions
            # Store the image index, transition index, and energy for this λ.
            lambda += 1
            image_of[lambda] = image_idx
            transition_of[lambda] = transition_idx
            lambda_of[image_idx, transition_idx] = lambda
            energies[lambda] = conformer_transition_energies[conformer_idx][transition_idx]
        end
    end

    return CrystalExcitationBasis{T}(
        image_of, transition_of, lambda_of, energies, zeros(T, n_lambda, n_lambda),
        images, n_transitions)
end

function set_D_couplings!(
        basis::CrystalExcitationBasis{T},
        blocks::AbstractArray{<:Real, 3},
    )::CrystalExcitationBasis{T} where {T<:AbstractFloat}
    """
    Store the same-molecule couplings D in the localised excitation basis.

    D only couples excitations on the same molecule, so it arrives as one block per image and is
    spread onto the (λ, λ') pairs here.

    # Arguments:
    - basis::CrystalExcitationBasis{T}: The basis whose D couplings are overwritten.
    - blocks::AbstractArray{<:Real,3}: D in eV, with dimensions (n_images, n_transitions, n_transitions).

    # Returns:
    - CrystalExcitationBasis{T}: The same basis, with its D updated.
    """

    # Get the dimensions of the D blocks.
    n_images = length(basis.images)
    n_transitions = basis.n_transitions
    size(blocks) == (n_images, n_transitions, n_transitions) || error(
        "The D blocks have size $(size(blocks)), expected $((n_images, n_transitions, n_transitions)).")
    all(isfinite, blocks) || error("The D blocks contain a non-finite value.")

    # D is real symmetric for the real densities RKS gives us, so check that each block is symmetric.
    for image in 1:n_images, t in 1:n_transitions, s in 1:t
        isapprox(blocks[image, s, t], blocks[image, t, s]; atol = sqrt(eps(T))) || error(
            "The D block of image $(image) is not symmetric at ($(s), $(t)): " *
            "$(blocks[image, s, t]) against $(blocks[image, t, s]).")
    end

    # Spread the D blocks onto the (λ, λ') pairs.
    fill!(basis.D_couplings, zero(T))
    for image in 1:n_images, t in 1:n_transitions, s in 1:n_transitions
        basis.D_couplings[basis.lambda_of[image, s], basis.lambda_of[image, t]] =
            T(blocks[image, s, t])
    end

    return basis
end

struct BlochEigensystem{T<:AbstractFloat, W, E}
    """
    The Bloch Hamiltonian at one k, together with its eigenvalues and eigenvectors.

    The whole thing is allocated once and overwritten at each q point, so that the inner loop over
    the momentum grid does not allocate.

    # Fields:
    - H::Matrix{Complex{T}}: The Bloch Hamiltonian at the current k.
    - coefficients::Matrix{Complex{T}}: The eigenvectors C, with column Ψ holding C_λ(Ψ).
    - energies::Vector{T}: The eigenvalues E_Ψ(k), in eV.
    - eigen_buffers::W: LAPACK buffers for the Hermitian eigensolve.
    - ewald_buffers::E: Buffers for the Ewald long-range sum, or nothing.
    - cell_phases::Vector{Complex{T}}: exp(i k . ΔR) for each neighbour cell at the current k.
    """
    H::Matrix{Complex{T}}
    coefficients::Matrix{Complex{T}}
    energies::Vector{T}
    eigen_buffers::W
    ewald_buffers::E
    cell_phases::Vector{Complex{T}}
end

function BlochEigensystem(
        basis::CrystalExcitationBasis{T},
        long_range::Union{Nothing, EwaldLongRangeData{T}} = nothing;
        n_cells::Int = 0,
        vecs::Bool = true,
    ) where {T<:AbstractFloat}
    """
    Allocate everything one task needs to build and diagonalise H(k), so the loop over k does not.

    Pass vecs = false when only the energies are wanted, as the band map does. This speeds up the
    diagonalisation when we don't need the eigenvectors.

    # Arguments:
    - basis::CrystalExcitationBasis{T}: The localised excitation basis, which sets n_lambda.
    - long_range::Union{Nothing, EwaldLongRangeData{T}}: The Ewald long-range data, or nothing for
      no reciprocal-space term.
    - n_cells::Int: The number of neighbour cells, sizing the cell phase table.
    - vecs::Bool: Whether the eigenvectors are needed as well as the energies.

    # Returns:
    - BlochEigensystem{T}: The buffers, sized for this basis.
    """

    # Get the number of states.
    n_lambda = length(basis.energies)

    # Allocate memory for the Hamiltonian.
    H = Matrix{Complex{T}}(undef, n_lambda, n_lambda)

    # If there are long-range terms, allocate buffers for them.
    ewald_buffers = long_range === nothing ? nothing : EwaldLongRangeBuffers(long_range)

    # Only allocate eigenvector storage if requested. This speeds up band map, where we only need energies.
    eigen_buffers = HermitianEigenWs(H; vecs = vecs)
    coefficients = vecs ? Matrix{Complex{T}}(undef, n_lambda, n_lambda) :
                          Matrix{Complex{T}}(undef, 0, 0)

    return BlochEigensystem{T, typeof(eigen_buffers), typeof(ewald_buffers)}(
        H,
        coefficients,
        Vector{T}(undef, n_lambda),
        eigen_buffers,
        ewald_buffers,
        Vector{Complex{T}}(undef, n_cells),
    )
end

@inline function build_bloch_hamiltonian!(
        eigensystem::BlochEigensystem{T},
        basis::CrystalExcitationBasis{T},
        k::SVector{3, T},
        couplings::Union{Nothing, CrystalCouplings{T}} = nothing,
        long_range::Union{Nothing, EwaldLongRangeData{T}} = nothing,
    ) where {T<:AbstractFloat}
    """
    Assemble the Bloch Hamiltonian

        H_{A_i s, B_j t}(k) = E_{A,s} δ_{AB} δ_{ij} δ_{st} + D_{A_i s, B_j t}
                              + Σ_{ΔR} J_{A_i s, B_j t}(ΔR) e^{i k . ΔR} + 𝒥^LR_{A_i s, B_j t}(k),

    where D couples excitations on the same molecule, so it is zero unless A_i = B_j, but it does
    mix different transitions s and t of that molecule.

    Setting D and both parts of J to zero leaves H diagonal and k-independent, which is the zeroth-order
    (incoherent) limit in which the crystal excitations are just the localised molecular ones. The couplings are split into the
    short- (J) and long-range (𝒥) parts using Ewald summation.

    # Arguments:
    - eigensystem::BlochEigensystem{T}: The eigensystem, whose H field is overwritten.
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - k::SVector{3, T}: The Brillouin zone wavevector, in Å^{-1}.
    - couplings::Union{Nothing, CrystalCouplings{T}}: The "short-range" J_{λλ'}(ΔR), or nothing for no coupling.
    - long_range::Union{Nothing, EwaldLongRangeData{T}}: The Ewald long-range data, or nothing when
      the couplings are the full unsplit lattice sum.

    # Returns:
    - Nothing. eigensystem.H is overwritten with the full Hermitian matrix.
    """

    # Zero the Hamiltonian.
    H = eigensystem.H
    fill!(H, zero(Complex{T}))

    # Diagonal: the monomer excitation energy.
    @inbounds for lambda in eachindex(basis.energies)
        H[lambda, lambda] = Complex{T}(basis.energies[lambda])
    end

    # D, which shifts each excitation and mixes those on the same molecule. Upper triangle only,
    # like J below, since the mirror at the end fills the rest in.
    @inbounds for lambda_prime in axes(basis.D_couplings, 2), lambda in 1:lambda_prime
        H[lambda, lambda_prime] += Complex{T}(basis.D_couplings[lambda, lambda_prime])
    end

    # Off-diagonal: the lattice-summed excitation transfer.
    # Only the upper triangle is accumulated below, which halves the work 
    # The lower triangle is then mirrored at the end to make H readable and avoid errors.
    # This also ensures exact Hermiticity.
    if couplings !== nothing
        n_lambda = length(basis.energies)
        n_cells = length(couplings.cell_vectors)
        phases = eigensystem.cell_phases
        # Sized on first use if the caller did not say how many cells there would be, so that
        # constructing the eigensystem without that hint still works rather than failing here.
        length(phases) == n_cells || resize!(phases, n_cells)
        @inbounds for cell_idx in 1:n_cells
            phases[cell_idx] = cis(dot(k, couplings.cell_vectors[cell_idx]))
        end

        # Σ_ΔR J(ΔR) exp(i k . ΔR).
        values = couplings.values
        @inbounds for lambda_prime in 1:n_lambda, lambda in 1:lambda_prime
            total = zero(Complex{T})
            @simd for cell_idx in 1:n_cells
                total += values[cell_idx, lambda, lambda_prime] * phases[cell_idx]
            end
            H[lambda, lambda_prime] += total
        end
    end

    # The Ewald long-range half, summed over Q = k + G. Also upper triangle only.
    if long_range !== nothing
        add_long_range!(H, long_range, eigensystem.ewald_buffers, k)
    end

    # Mirror the upper triangle into the lower one.
    n_lambda = length(basis.energies)
    @inbounds for lambda_prime in 1:n_lambda, lambda in 1:(lambda_prime - 1)
        H[lambda_prime, lambda] = conj(H[lambda, lambda_prime])
    end

    return nothing
end

@inline function solve_bloch_hamiltonian!(
        eigensystem::BlochEigensystem{T},
        basis::CrystalExcitationBasis{T},
        k::SVector{3, T},
        couplings::Union{Nothing, CrystalCouplings{T}} = nothing,
        long_range::Union{Nothing, EwaldLongRangeData{T}} = nothing,
    ) where {T<:AbstractFloat}
    """
    Build and diagonalise the Bloch Hamiltonian at a single k, giving the crystal excitation
    energies E_Ψ(k) and the coefficients C_{A_i,s}^Ψ(k) that coherently mix the localised molecular
    excitations.

    ## The state labelling convention:

    Ψ is defined by ascending energy: Ψ = 1 is the lowest crystal excitation at this k, Ψ = 2 the
    next, and so on. This gives a canonical labelling of the branches for later reprodcability, if we
    only have H.

    # Arguments:
    - eigensystem::BlochEigensystem{T}: The eigensystem, whose fields are overwritten.
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - k::SVector{3, T}: The Brillouin zone wavevector, in Å^{-1}.
    - couplings::Union{Nothing, CrystalCouplings{T}}: The "short-range" J_{λλ'}(ΔR) couplings, or nothing for no coupling.
    - long_range::Union{Nothing, EwaldLongRangeData{T}}: The Ewald long-range data, or nothing.

    # Returns:
    - Nothing. eigensystem.energies and eigensystem.coefficients are modified in place, with column
      Ψ of the coefficients holding C_λ(Ψ), and Ψ ordered by ascending energy.
    """

    build_bloch_hamiltonian!(eigensystem, basis, k, couplings, long_range)


    # Use syevr! to diagonalise the Hamiltonian, which overwrites rather than allocates. 
    #Eigevalues are also returned in ascending order, matching our convention.
    values, vectors = LinearAlgebra.LAPACK.syevr!(
        eigensystem.eigen_buffers, 'V', 'A', 'U', eigensystem.H,
        zero(T), zero(T), 0, 0, -one(T),
    )

    # Store the energies and coefficients.
    eigensystem.energies .= values
    eigensystem.coefficients .= vectors

    return nothing
end

@inline function solve_bloch_energies!(
        eigensystem::BlochEigensystem{T},
        basis::CrystalExcitationBasis{T},
        k::SVector{3, T},
        couplings::Union{Nothing, CrystalCouplings{T}} = nothing,
        long_range::Union{Nothing, EwaldLongRangeData{T}} = nothing,
    ) where {T<:AbstractFloat}
    """
    Build H(k) and return its eigenvalues alone, for callers that never look at the coefficients.

    Asking syevr for 'N' rather than 'V' skips eigenvector calculation.

    # Arguments:
    - eigensystem::BlochEigensystem{T}: The eigensystem, whose H and energies are overwritten.
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - k::SVector{3, T}: The Brillouin zone wavevector, in Å^{-1}.
    - couplings::Union{Nothing, CrystalCouplings{T}}: The short-range couplings, or nothing.
    - long_range::Union{Nothing, EwaldLongRangeData{T}}: The Ewald long-range data, or nothing.

    # Returns:
    - Nothing. Only eigensystem.energies is modified, in ascending order.
    """

    build_bloch_hamiltonian!(eigensystem, basis, k, couplings, long_range)

    values, _ = LinearAlgebra.LAPACK.syevr!(
        eigensystem.eigen_buffers, 'N', 'A', 'U', eigensystem.H,
        zero(T), zero(T), 0, 0, -one(T),
    )
    eigensystem.energies .= values

    return nothing
end

end
