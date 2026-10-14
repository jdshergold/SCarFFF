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
    The separation-resolved intermolecular couplings J_{λ_iλ_j'}(ΔR), and the lattice vectors they are indexed by.

    # Fields:
    Under Ewald these contain J_SR(ΔR) - δ_{ΔR,0} J^self, whose Bloch sum
    combines with J_bar^LR(k). Under direct summation they contain the bare J(ΔR).

    - J_deltaR::Array{Complex{T}, 3}: The couplings in eV, with dimensions (n_neighbour_cells, n_lambda_i, n_lambda_i).
    - Delta_R::Vector{SVector{3, T}}: The lattice vector ΔR for each cell, in Å.
    """
    J_deltaR::Array{Complex{T}, 3}
    Delta_R::Vector{SVector{3, T}}
end

struct CrystalImage{T<:AbstractFloat}
    """
    One molecule in the unit cell: the (A, i) label of the theory.

    # Fields:
    - conformer_index::Int: Which reference conformer (A) this is an image of.
    - R_tilde::Quaternionic.Rotor{T}: The proper rotation R̃_{A,i} mapping the reference monomer onto this image.
    - det_rotation::T: det(R_{A,i}), which is -1 for improper operations.
    - tau::SVector{3, T}: The translation τ_{A,i} from the cell origin to this image, in Å.
    """
    conformer_index::Int
    R_tilde::Quaternionic.Rotor{T}
    det_rotation::T
    tau::SVector{3, T}
end

struct CrystalExcitationBasis{T<:AbstractFloat}
    """
    The localised excitation basis λ_i = (A, i, s), flattened to a single index.

    The ordering defined here is used by both the Bloch Hamiltonian and the
    coherent form factor sum, so that the eigenvector components line up with the molecular
    amplitudes. It is image-major, so λ_i runs over transitions fastest, then over images.

    For example, image_of[λ_i] gives the image index (A, i) for each λ_i, and transition_of[λ_i] gives the transition index s for each λ_i.
    In a unit cell with say, 2 molecules, and 3 transitions, this would have 6 entries.
    They would be ordered as follows:

    λ_i = 1-3: image 1, transition 1-3,
    λ_i = 4-6: image 2, transition 1-3.

    To get the actual information about a specific image, you would use something like images[image_of[λ_i]].

    # Fields:
    - image_of::Vector{Int}: The image index (A, i) for each λ_i.
    - transition_of::Vector{Int}: The transition index s for each λ_i, as a position in the requested transition list.
    - lambda_i_of::Matrix{Int}: The λ_i carrying each (image, transition), which inverts the two above.
    - Delta_E_monomer::Vector{T}: The monomer excitation energy ΔE_λ_i for each λ_i, in eV.
    - D_couplings::Matrix{T}: The same-molecule couplings D_{λ_iλ_j′} in eV, non-zero only for λ_i and λ_j′
      on the same image.
    - images::Vector{CrystalImage{T}}: The images themselves.
    - n_transitions::Int: The number of transitions per image.
    """
    image_of::Vector{Int}
    transition_of::Vector{Int}
    lambda_i_of::Matrix{Int}
    Delta_E_monomer::Vector{T}
    D_couplings::Matrix{T}
    images::Vector{CrystalImage{T}}
    n_transitions::Int
end

function build_excitation_basis(
        images::Vector{CrystalImage{T}},
        conformer_transition_energies::Vector{Vector{T}},
    )::CrystalExcitationBasis{T} where {T<:AbstractFloat}
    """
    Flatten the localised excitations (A, i, s) into a single index λ_i.

    # Arguments:
    - images::Vector{CrystalImage{T}}: The molecules in the unit cell.
    - conformer_transition_energies::Vector{Vector{T}}: Excitation energies in eV, indexed [conformer][transition].

    # Returns:
    - CrystalExcitationBasis{T}: The flattened basis.
    """

    isempty(images) && error("Cannot build an excitation basis with no molecules in the unit cell.")

    n_transitions = length(conformer_transition_energies[1])
    # Check that all conformers have the same number of transitions.
    for (conformer_idx, Delta_E_monomer) in enumerate(conformer_transition_energies)
        length(Delta_E_monomer) == n_transitions ||
            error("Conformer $(conformer_idx) has $(length(Delta_E_monomer)) transitions, but conformer 1 has $(n_transitions). All conformers must be computed for the same transitions.")
    end

    # Allocate the arrays for the flattened basis. 
    n_lambda_i = length(images) * n_transitions
    image_of = Vector{Int}(undef, n_lambda_i)
    transition_of = Vector{Int}(undef, n_lambda_i)
    lambda_i_of = Matrix{Int}(undef, length(images), n_transitions)
    Delta_E_monomer = Vector{T}(undef, n_lambda_i)

    lambda_i = 0
    for (image_idx, image) in enumerate(images)
        conformer_idx = image.conformer_index
        # Check that the conformer index is valid and that there are transition energies for it.
        conformer_idx in eachindex(conformer_transition_energies) ||
            error("Image $(image_idx) refers to conformer $(conformer_idx), which has no transition energies.")

        for transition_idx in 1:n_transitions
            # Store the image index, transition index, and energy for this λ_i.
            lambda_i += 1
            image_of[lambda_i] = image_idx
            transition_of[lambda_i] = transition_idx
            lambda_i_of[image_idx, transition_idx] = lambda_i
            Delta_E_monomer[lambda_i] = conformer_transition_energies[conformer_idx][transition_idx]
        end
    end

    return CrystalExcitationBasis{T}(
        image_of, transition_of, lambda_i_of, Delta_E_monomer, zeros(T, n_lambda_i, n_lambda_i),
        images, n_transitions)
end

function set_D_couplings!(
        basis::CrystalExcitationBasis{T},
        blocks::AbstractArray{<:Real, 3},
    )::CrystalExcitationBasis{T} where {T<:AbstractFloat}
    """
    Store the same-molecule couplings D in the localised excitation basis.

    D only couples excitations on the same molecule, so it arrives as one block per image and is
    spread onto the (λ_i, λ_j′) pairs here.

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

    # Spread the D blocks onto the (λ_i, λ_j′) pairs.
    fill!(basis.D_couplings, zero(T))
    for image in 1:n_images, t in 1:n_transitions, s in 1:n_transitions
        basis.D_couplings[basis.lambda_i_of[image, s], basis.lambda_i_of[image, t]] =
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
    - C::Matrix{Complex{T}}: The eigenvectors C, with column n holding C_{λ_i}^{(n)}(k).
    - Delta_E::Vector{T}: The eigenvalues ΔE_n(k), in eV.
    - eigen_buffers::W: LAPACK buffers for the Hermitian eigensolve.
    - ewald_buffers::E: Buffers for the Ewald long-range sum, or nothing.
    - cell_phases::Vector{Complex{T}}: exp(i k . ΔR) for each neighbour cell at the current k.
    """
    H::Matrix{Complex{T}}
    C::Matrix{Complex{T}}
    Delta_E::Vector{T}
    eigen_buffers::W
    ewald_buffers::E
    cell_phases::Vector{Complex{T}}
end

function BlochEigensystem(
        basis::CrystalExcitationBasis{T},
        long_range::Union{Nothing, EwaldLongRangeData{T}} = nothing;
        n_neighbour_cells::Int = 0,
        vecs::Bool = true,
    ) where {T<:AbstractFloat}
    """
    Allocate everything one task needs to build and diagonalise H(k), so the loop over k does not.

    Pass vecs = false when only the excitation energies are wanted, as the band map does. This speeds up the
    diagonalisation when we don't need the eigenvectors.

    # Arguments:
    - basis::CrystalExcitationBasis{T}: The localised excitation basis, which sets n_lambda_i.
    - long_range::Union{Nothing, EwaldLongRangeData{T}}: The Ewald long-range data, or nothing for
      no reciprocal-space term.
    - n_neighbour_cells::Int: The number of neighbour cells, sizing the cell phase table.
    - vecs::Bool: Whether the eigenvectors are needed as well as the excitation energies.

    # Returns:
    - BlochEigensystem{T}: The buffers, sized for this basis.
    """

    # Get the number of states.
    n_lambda_i = length(basis.Delta_E_monomer)

    # Allocate memory for the Hamiltonian.
    H = Matrix{Complex{T}}(undef, n_lambda_i, n_lambda_i)

    # If there are long-range terms, allocate buffers for them.
    ewald_buffers = long_range === nothing ? nothing : EwaldLongRangeBuffers(long_range)

    # Only allocate eigenvector storage if requested. This speeds up band map, where we only need Delta_E.
    eigen_buffers = HermitianEigenWs(H; vecs = vecs)
    C = vecs ? Matrix{Complex{T}}(undef, n_lambda_i, n_lambda_i) :
                          Matrix{Complex{T}}(undef, 0, 0)

    return BlochEigensystem{T, typeof(eigen_buffers), typeof(ewald_buffers)}(
        H,
        C,
        Vector{T}(undef, n_lambda_i),
        eigen_buffers,
        ewald_buffers,
        Vector{Complex{T}}(undef, n_neighbour_cells),
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

        H_{A_i s, B_j t}(k) = ΔE_λ_i δ_{AB} δ_{ij} δ_{st} + D_{A_i s, B_j t}
                              + J^SR_{A_i s, B_j t}(k) + J^LR_{A_i s, B_j t}(k),

    with J^LR(k) = J_bar^LR(k) - J^self. The code stores the self subtraction in
    couplings.J_deltaR, as J_SR(ΔR) - δ_{ΔR,0} J^self, and adds J_bar^LR(k) afterwards.

    D couples excitations on the same molecule, so it is zero unless A_i = B_j, but it does
    mix different transitions s and t of that molecule.

    Setting D and both parts of J to zero leaves H diagonal and k-independent, which is the zeroth-order
    (incoherent) limit in which the crystal excitations are just the localised molecular ones. The couplings are split into the
    short- (J^SR) and long-range (J^LR) parts using Ewald summation.

    # Arguments:
    - eigensystem::BlochEigensystem{T}: The eigensystem, whose H field is overwritten.
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - k::SVector{3, T}: The Brillouin zone wavevector, in Å^{-1}.
    - couplings::Union{Nothing, CrystalCouplings{T}}: The stored J_deltaR couplings, or nothing for no coupling.
    - long_range::Union{Nothing, EwaldLongRangeData{T}}: The Ewald long-range data, or nothing when
      the couplings are the full unsplit lattice sum.

    # Returns:
    - Nothing. eigensystem.H is overwritten with the full Hermitian matrix.
    """

    # Zero the Hamiltonian.
    H = eigensystem.H
    fill!(H, zero(Complex{T}))

    # Diagonal: the monomer excitation energy.
    @inbounds for lambda_i in eachindex(basis.Delta_E_monomer)
        H[lambda_i, lambda_i] = Complex{T}(basis.Delta_E_monomer[lambda_i])
    end

    # D, which shifts each excitation and mixes those on the same molecule. Upper triangle only,
    # like J below, since the mirror at the end fills the rest in.
    @inbounds for lambda_j in axes(basis.D_couplings, 2), lambda_i in 1:lambda_j
        H[lambda_i, lambda_j] += Complex{T}(basis.D_couplings[lambda_i, lambda_j])
    end

    # Off-diagonal: the lattice-summed excitation transfer.
    # Only the upper triangle is accumulated below, which halves the work 
    # The lower triangle is then mirrored at the end to make H readable and avoid errors.
    # This also ensures exact Hermiticity.
    if couplings !== nothing
        n_lambda_i = length(basis.Delta_E_monomer)
        n_neighbour_cells = length(couplings.Delta_R)
        phases = eigensystem.cell_phases
        # Sized on first use if the caller did not say how many cells there would be, so that
        # constructing the eigensystem without that hint still works rather than failing here.
        length(phases) == n_neighbour_cells || resize!(phases, n_neighbour_cells)
        @inbounds for cell_idx in 1:n_neighbour_cells
            phases[cell_idx] = cis(dot(k, couplings.Delta_R[cell_idx]))
        end

        # Σ_ΔR J(ΔR) exp(i k . ΔR).
        values = couplings.J_deltaR
        @inbounds for lambda_j in 1:n_lambda_i, lambda_i in 1:lambda_j
            total = zero(Complex{T})
            @simd for cell_idx in 1:n_neighbour_cells
                total += values[cell_idx, lambda_i, lambda_j] * phases[cell_idx]
            end
            H[lambda_i, lambda_j] += total
        end
    end

    # The Ewald long-range half, summed over Q = k + G. Also upper triangle only.
    if long_range !== nothing
        add_long_range!(H, long_range, eigensystem.ewald_buffers, k)
    end

    # Mirror the upper triangle into the lower one.
    n_lambda_i = length(basis.Delta_E_monomer)
    @inbounds for lambda_j in 1:n_lambda_i, lambda_i in 1:(lambda_j - 1)
        H[lambda_j, lambda_i] = conj(H[lambda_i, lambda_j])
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
    energies ΔE_n(k) and the coefficients C_{λ_i}^{(n)}(k) that coherently mix the localised molecular
    excitations.

    ## The state labelling convention:

    n is defined by ascending energy: n = 1 is the lowest crystal excitation at this k, n = 2 the
    next, and so on. This gives a canonical labelling of the branches for later reprodcability, if we
    only have H.

    # Arguments:
    - eigensystem::BlochEigensystem{T}: The eigensystem, whose fields are overwritten.
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - k::SVector{3, T}: The Brillouin zone wavevector, in Å^{-1}.
    - couplings::Union{Nothing, CrystalCouplings{T}}: The "short-range" J_{λ_iλ_j′}(ΔR) couplings, or nothing for no coupling.
    - long_range::Union{Nothing, EwaldLongRangeData{T}}: The Ewald long-range data, or nothing.

    # Returns:
    - Nothing. eigensystem.Delta_E and eigensystem.C are modified in place, with column
      n of the coefficients holding C_{λ_i}^{(n)}(k), and n ordered by ascending energy.
    """

    build_bloch_hamiltonian!(eigensystem, basis, k, couplings, long_range)


    # Use syevr! to diagonalise the Hamiltonian, which overwrites rather than allocates. 
    #Eigevalues are also returned in ascending order, matching our convention.
    values, vectors = LinearAlgebra.LAPACK.syevr!(
        eigensystem.eigen_buffers, 'V', 'A', 'U', eigensystem.H,
        zero(T), zero(T), 0, 0, -one(T),
    )

    # Store the energies and coefficients.
    eigensystem.Delta_E .= values
    eigensystem.C .= vectors

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
    - Nothing. Only eigensystem.Delta_E is modified, in ascending order.
    """

    build_bloch_hamiltonian!(eigensystem, basis, k, couplings, long_range)

    values, _ = LinearAlgebra.LAPACK.syevr!(
        eigensystem.eigen_buffers, 'N', 'A', 'U', eigensystem.H,
        zero(T), zero(T), 0, 0, -one(T),
    )
    eigensystem.Delta_E .= values

    return nothing
end

end
