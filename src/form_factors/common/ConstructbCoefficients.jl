# This module contains functions to compute the b_{ij}^{A/B/C} coefficients.

module ConstructbCoefficients

using StaticArrays

using ..ReadBasisSet: MoleculeData

export construct_b_coefficients

# Precompute binomial coefficients for up to n = 6, corresponding to f-orbitals, for speed.
const binomial_cache = SMatrix{7, 7, Int}(binomial(n, k) for n in 0:6, k in 0:6)

@inline function get_b_ij_A(A::Int, a_i::Int, a_j::Int, X_ij::T, x_i::T, x_j::T)::T where {T<:AbstractFloat}
    """
    Compute a single b_{ij}^A coefficient for the pair of Cartesian terms i and j, at a given A.

    # Arguments:
    - A::Int: The A index for the b_{ij}^A coefficient.
    - a_i::Int: The power of x in Cartesian term i.
    - a_j::Int: The power of x in Cartesian term j.
    - X_ij::T: The x-component of the R_{ij} vector.
    - x_i::T: The x-coordinate of the primitive corresponding to Cartesian term i.
    - x_j::T: The x-coordinate of the primitive corresponding to Cartesian term j.
    # Returns:
    - b_ij_A::T: The computed b_{ij}^A coefficient.
    """

    b_ij_A = zero(T)

    # Find the limits of the sum.
    k_min = max(0, a_i - A)
    k_max = min(a_i, a_i + a_j - A)

    # Accumulate the sum for b_{ij}^A.
    for k in k_min:k_max
        binom_i = binomial_cache[a_i + 1, k + 1]
        binom_j = binomial_cache[a_j + 1, a_i + a_j - k - A + 1]

        term = binom_i * binom_j * (X_ij - x_i)^k * (X_ij - x_j)^(a_i + a_j - k - A)

        b_ij_A += term
    end

    return b_ij_A
end

function construct_b_coefficients(mol::MoleculeData{T}, R_ij::Array{T, 3})::Tuple{Array{T, 3}, Array{T, 3}, Array{T, 3}} where {T<:AbstractFloat}
    """
    Construct the b_{ij}^{A/B/C} coefficients for all Cartesian pairs of primitives in the molecule.
    These are defined by:

        b_{ij}^A = ∑_{k = max{0, a_i - A}}^{min{a_i, a_i + a_j - A}} binom(a_i, k) binom(a_j, a_i + a_j - k - A)
                 * (X_ij - x_i)^k (X_ij - x_j)^(a_i + a_j - k - A),

    where A ranges from 0 to a_i + a_j, a_i and a_j are the powers of x in the Cartesian terms i and j,
    X_ij is the x-component of R_ij, and x_i and x_j are the x-coordinates of the primitives corresponding
    to Cartesian terms i and j, respectively. Similar definitions hold for b_{ij}^B and b_{ij}^C in the y2
    and z-directions, respectively.

    # Arguments:
    - mol::MoleculeData : The MoleculeData structure containing the molecule and basis set information.
    - R_ij::Array{T,3}: The R_{ij} vectors for all Cartesian term pairs.

    - Returns:
    - b_A::Array{T,3}: The b_{ij}^A coefficients for all Cartesian term pairs.
    - b_B::Array{T,3}: The b_{ij}^B coefficients for all Cartesian term pairs.
    - b_C::Array{T,3}: The b_{ij}^C coefficients for all Cartesian term pairs.
    """

    n_cartesian_terms = mol.n_cartesian_terms

    # Compute the maximum A, B and C values for the full set.
    max_ABC = 2 * maximum(mol.orbital_l)

    # Allocate memory for the b coefficients. Initialise them as zeros so that those with A > a_i + a_j are already zero.
    b_A = zeros(T, n_cartesian_terms, n_cartesian_terms, max_ABC + 1)
    b_B = zeros(T, n_cartesian_terms, n_cartesian_terms, max_ABC + 1)
    b_C = zeros(T, n_cartesian_terms, n_cartesian_terms, max_ABC + 1)

    @inbounds for i in 1:n_cartesian_terms
        atom_idx_i = mol.cartesian_term_to_atom[i]

        a_i = mol.cartesian_a[i]
        b_i = mol.cartesian_b[i]
        c_i = mol.cartesian_c[i]

        x_i = mol.atom_coordinates[atom_idx_i, 1]
        y_i = mol.atom_coordinates[atom_idx_i, 2]
        z_i = mol.atom_coordinates[atom_idx_i, 3]

        # These coefficients are symmetric in i and j, so only compute for j >= i.
        @inbounds for j in i:n_cartesian_terms
            atom_idx_j = mol.cartesian_term_to_atom[j]

            a_j = mol.cartesian_a[j]
            b_j = mol.cartesian_b[j]
            c_j = mol.cartesian_c[j]

            x_j = mol.atom_coordinates[atom_idx_j, 1]
            y_j = mol.atom_coordinates[atom_idx_j, 2]
            z_j = mol.atom_coordinates[atom_idx_j, 3]

            X_ij = R_ij[i, j, 1]
            Y_ij = R_ij[i, j, 2]
            Z_ij = R_ij[i, j, 3]

            # Compute b_{ij}^A coefficients.
            @inbounds for A in 0:(a_i + a_j)
                b_A[i, j, A + 1] = get_b_ij_A(A, a_i, a_j, X_ij, x_i, x_j)
                b_A[j, i, A + 1] = b_A[i, j, A + 1] # Symmetry.
            end

            # Compute b_{ij}^B coefficients.
            @inbounds for B in 0:(b_i + b_j)
                b_B[i, j, B + 1] = get_b_ij_A(B, b_i, b_j, Y_ij, y_i, y_j)
                b_B[j, i, B + 1] = b_B[i, j, B + 1]
            end

            # Compute b_{ij}^C coefficients.
            @inbounds for C in 0:(c_i + c_j)
                b_C[i, j, C + 1] = get_b_ij_A(C, c_i, c_j, Z_ij, z_i, z_j)
                b_C[j, i, C + 1] = b_C[i, j, C + 1]
            end
        end
    end

    return b_A, b_B, b_C
end
end
