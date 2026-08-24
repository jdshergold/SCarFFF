# This module contains the lattice geometry needed by the crystal excitation treatment:
# the reciprocal lattice, and the folding of a momentum transfer q into the first Brillouin zone.

module CrystalLattice

using StaticArrays
using LinearAlgebra: det, inv

export CrystalLatticeData, build_crystal_lattice, fold_to_bz,
       KEV_TO_INV_ANGSTROM, ALPHA_EM, HBAR_C_EV_ANGSTROM

const KEV_TO_INV_ANGSTROM = 1.0 / 1.973269804  # Multiplicative factor to convert keV to inverse Å.

# The fine structure constant, and the conversion from inverse Angstroms to eV in natural units.
const ALPHA_EM = 1.0 / 137.035999084
const HBAR_C_EV_ANGSTROM = 1973.269804

struct CrystalLatticeData{T<:AbstractFloat}
    direct::SMatrix{3, 3, T, 9}      # Rows are the direct lattice vectors a_1, a_2, a_3, in Å.
    reciprocal::SMatrix{3, 3, T, 9}  # Rows are the reciprocal lattice vectors b_1, b_2, b_3, in Å^{-1}.
    volume::T                        # The unit cell volume, in Å^3.
end

function build_crystal_lattice(direct_lattice::AbstractMatrix, ::Type{T}) where {T<:AbstractFloat}
    """
    Construct the lattice data from the direct lattice vectors, including the reciprocal lattice
    defined by the standard convention:

        a_i . b_j = 2π δ_{ij}.

    Writing A for the matrix whose rows are the a_i and B for the matrix whose rows are the b_j,
    that condition is A Bᵀ = 2π I, so B = 2π (A^{-1})ᵀ.

    # Arguments:
    - direct_lattice::AbstractMatrix: The 3x3 direct lattice, with rows a_1, a_2, a_3 in Å.
    - T::Type: The floating point type to use.

    # Returns:
    - CrystalLatticeData{T}: The direct and reciprocal lattices, and the cell volume.
    """

    size(direct_lattice) == (3, 3) || error("The direct lattice must be a 3x3 matrix, got $(size(direct_lattice)).")

    A = SMatrix{3, 3, T, 9}(T.(direct_lattice))
    volume = abs(det(A))

    # A degenerate cell would make the reciprocal lattice meaningless, so catch it here rather than
    # letting it surface as a silently wrong inverse.
    volume > eps(T) || error("The direct lattice is singular (volume $(volume) Å^3), so the reciprocal lattice is undefined.")

    B = T(2π) * transpose(inv(A))

    return CrystalLatticeData{T}(A, B, volume)
end

@inline function fold_to_bz(lattice::CrystalLatticeData{T}, q::SVector{3, T})::Tuple{SVector{3, T}, SVector{3, T}} where {T<:AbstractFloat}
    """
    Split a momentum transfer into a Brillouin zone wavevector and a reciprocal lattice vector,

        q = k + G,

    which is the selection rule that survives the lattice sum in the crystal form factor.

    The fractional coordinates of q in the reciprocal basis are x_i = (a_i . q) / 2π, which follows
    from q = Σ_i x_i b_i together with a_i . b_j = 2π δ_{ij}. Rounding each x_i to the nearest
    integer gives G, and the remainder gives k.

    This puts k in the parallelepiped fundamental domain x_i ∈ [-1/2, 1/2), rather than the
    Wigner-Seitz cell that is usually drawn as "the" first Brillouin zone. Either is a valid choice:
    the two differ only by reciprocal lattice vectors, and every k-dependent quantity here is
    periodic under k -> k + G (the Bloch Hamiltonian because G . ΔR ∈ 2πZ for lattice vectors ΔR),
    so the physics is unchanged. The parallelepiped is used because it is exact and needs no
    neighbour search.

    # Arguments:
    - lattice::CrystalLatticeData{T}: The lattice data.
    - q::SVector{3, T}: The momentum transfer, in Å^{-1}.

    # Returns:
    - k::SVector{3, T}: The Brillouin zone wavevector, in Å^{-1}.
    - G::SVector{3, T}: The reciprocal lattice vector, in Å^{-1}.
    """

    # Fractional coordinates of q in the reciprocal basis, x_i = (a_i . q) / 2π.
    fractional = (lattice.direct * q) ./ T(2π)

    # Split into the integer part (which gives G) and the remainder (which gives k).
    integer_part = round.(fractional)

    # Reconstruct G from its integer coordinates, and take k as the remainder. We form k by
    # subtraction rather than from its own fractional coordinates so that k + G reproduces q exactly.
    G = transpose(lattice.reciprocal) * integer_part
    k = q - G

    return k, G
end

end
