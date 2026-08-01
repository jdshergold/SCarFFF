# This module builds and diagonalises the Frenkel exciton Bloch Hamiltonian for a crystal.

module BlochHamiltonian

using StaticArrays
using Quaternionic
using LinearAlgebra: Hermitian, eigen!

export CrystalImage, CrystalExcitationBasis, build_excitation_basis,
       build_bloch_hamiltonian!, solve_bloch_hamiltonian!, BlochEigensystem

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

    The ordering defined here is the one contract used by both the Bloch Hamiltonian and the
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
    - energies::Vector{T}: The monomer excitation energy E_{A,s} for each λ, in eV.
    - images::Vector{CrystalImage{T}}: The images themselves.
    - n_transitions::Int: The number of transitions per image.
    """
    image_of::Vector{Int}
    transition_of::Vector{Int}
    energies::Vector{T}
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
            energies[lambda] = conformer_transition_energies[conformer_idx][transition_idx]
        end
    end

    return CrystalExcitationBasis{T}(image_of, transition_of, energies, images, n_transitions)
end

struct BlochEigensystem{T<:AbstractFloat}
    """
    The Bloch Hamiltonian at one k, together with its eigenvalues and eigenvectors.

    The whole thing is allocated once and overwritten at each q point, so that the inner loop over
    the momentum grid does not allocate.

    # Fields:
    - H::Matrix{Complex{T}}: The Bloch Hamiltonian at the current k.
    - coefficients::Matrix{Complex{T}}: The eigenvectors C, with column Ψ holding C_λ(Ψ).
    - energies::Vector{T}: The eigenvalues E_Ψ(k), in eV.
    """
    H::Matrix{Complex{T}}
    coefficients::Matrix{Complex{T}}
    energies::Vector{T}
end

function BlochEigensystem(basis::CrystalExcitationBasis{T}) where {T<:AbstractFloat}
    n_lambda = length(basis.energies)
    return BlochEigensystem{T}(
        Matrix{Complex{T}}(undef, n_lambda, n_lambda),
        Matrix{Complex{T}}(undef, n_lambda, n_lambda),
        Vector{T}(undef, n_lambda),
    )
end

@inline function build_bloch_hamiltonian!(
        eigensystem::BlochEigensystem{T},
        basis::CrystalExcitationBasis{T},
        k::SVector{3, T},
    ) where {T<:AbstractFloat}
    """
    Assemble the Bloch Hamiltonian

        H_{A_i s, B_j t}(k) = (E_{A,s} + D_{A_i,s}) δ_{AB} δ_{ij} δ_{st} + Σ_{ΔR} J_{A_i s, B_j t}(ΔR) e^{i k . ΔR}.

    Phase 2a of the crystal upgrade sets the intermolecular couplings to zero, so D = 0 and J = 0
    and H is simply the diagonal matrix of monomer excitation energies, independent of k. The full
    matrix and the general eigensolve are still built and used, so that turning the couplings on in
    phase 2b is a matter of filling in the two terms marked below and nothing else.

    Note that H is periodic under k -> k + G for any reciprocal lattice vector G, since G . ΔR ∈ 2πZ
    for any lattice vector ΔR. Any Brillouin zone representative therefore gives the same result.

    # Arguments:
    - eigensystem::BlochEigensystem{T}: The eigensystem, whose H field is overwritten.
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - k::SVector{3, T}: The Brillouin zone wavevector, in Å^{-1}.

    # Returns:
    - Nothing. eigensystem.H is modified in place.
    """

    H = eigensystem.H
    fill!(H, zero(Complex{T}))

    # Diagonal: the monomer excitation energy, plus the D_{A_i,s} environment shift.
    # TODO: add D_{A_i,s}, the Coulomb interaction of this molecule's excited-minus-ground
    # difference density with the ground state density and nuclei of every other molecule.
    @inbounds for lambda in eachindex(basis.energies)
        H[lambda, lambda] = Complex{T}(basis.energies[lambda])
    end

    # Off-diagonal: the lattice-summed excitation transfer.
    # TODO: add Σ_{ΔR} J_{A_i s, B_j t}(ΔR) exp(i k . ΔR), with J the Coulomb coupling
    # between transition densities and the sum truncated at a real-space neighbour cutoff.
    # J_{A_i s, A_i s}(0) is excluded, as that contribution already sits in the diagonal energy.

    return nothing
end

@inline function solve_bloch_hamiltonian!(
        eigensystem::BlochEigensystem{T},
        basis::CrystalExcitationBasis{T},
        k::SVector{3, T},
    ) where {T<:AbstractFloat}
    """
    Build and diagonalise the Bloch Hamiltonian at a single k, giving the crystal excitation
    energies E_Ψ(k) and the coefficients C_{A_i,s}^Ψ(k) that coherently mix the localised molecular
    excitations.

    ## The state labelling convention

    Ψ is defined by ascending energy: Ψ = 1 is the lowest crystal excitation at this k, Ψ = 2 the
    next, and so on. This is nice for the following reasons:

    - It is canonical. The n-th lowest eigenvalue is a property of H alone, so it is reproducible on
      any machine and at any later time. Rebuilding H from the stored couplings and re-diagonalising
      recovers exactly the same assignment, which is what makes it safe to store the form factor and
      the energies separately and pair them up again afterwards.
    - It is continuous. Eigenvalues depend continuously on the matrix and sorting preserves that, so
      the sorted branches are continuous in k without any band-tracking machinery.

    That second point is worth unpacking, because the obvious worry is a fair one: if we diagonalise
    independently at every k, what stops the label Ψ = 3 from meaning one branch at one k and a
    different branch at the next?

    The answer is that two branches can only touch if three separate conditions hold at once. Take
    the 2x2 Hermitian block spanned by the two states in question,

        [ a   c  ]
        [ c*  b  ],

    whose eigenvalues coincide only when a = b, Re(c) = 0 and Im(c) = 0. That is three constraints,
    and k-space has only three dimensions to satisfy them in, which leaves the solutions as isolated
    points rather than lines or surfaces. A path through k therefore essentially never lands on one.
    What it meets instead is an avoided crossing: the two branches approach, exchange which molecules
    dominate them, and separate again without ever meeting. The composition of a state changes there,
    but its identity as "the third lowest" does not.

    The exception is a degeneracy imposed by crystal symmetry. Symmetry can force the three
    conditions to hold along an entire high-symmetry line rather than at isolated points, so the
    branches genuinely do meet. The labelling stays well defined even then, but the branches meet in
    a cone, which makes them continuous without being smooth.

    # Arguments:
    - eigensystem::BlochEigensystem{T}: The eigensystem, whose fields are overwritten.
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - k::SVector{3, T}: The Brillouin zone wavevector, in Å^{-1}.

    # Returns:
    - Nothing. eigensystem.energies and eigensystem.coefficients are modified in place, with column
      Ψ of the coefficients holding C_λ(Ψ), and Ψ ordered by ascending energy.
    """

    build_bloch_hamiltonian!(eigensystem, basis, k)

    # eigen! overwrites its argument, which is why H is rebuilt from scratch on every call. For a
    # Hermitian argument it returns eigenvalues in ascending order, with matching eigenvector
    # columns, which is exactly the Ψ ordering described above.
    factorisation = eigen!(Hermitian(eigensystem.H))

    # Store the energies and coefficients.
    eigensystem.energies .= factorisation.values
    eigensystem.coefficients .= factorisation.vectors

    return nothing
end

end
