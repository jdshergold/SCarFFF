# This module contains functions to construct all quantities that depend only on Cartesian pairs.

module ConstructPairCoefficients

using ..ReadBasisSet: MoleculeData

export construct_pair_coefficients

@inline function get_sigma_bar_ij(sigma_i::T, sigma_j::T)::T where {T<:AbstractFloat}
    """ 
    Compute the quantity:

        σb_{ij} = √(σ_i^2 + σ_j^2),

    where σ_i and σ_j are the widths corresponding to Cartesian terms i and j, respectively.
    All inputs and outputs are given in units of Angstroms.

    # Arguments:
    - sigma_i::T: Width of primitive corresponding to Cartesian term i.
    - sigma_j::T: Width of primitive corresponding to Cartesian term j.

    # Returns:
    - sigma_ij_bar::T: The computed σb_{ij} value.
    """

    return sqrt(sigma_i * sigma_i + sigma_j * sigma_j)
end

@inline function get_sigma_ij(sigma_i::T, sigma_j::T)::T where {T<:AbstractFloat}
    """
    Compute the quantity:

        σ_{ij} = (σ_i * σ_j) / √(σ_i^2 + σ_j^2),

    where σ_i and σ_j are the widths corresponding to Cartesian terms i and j, respectively.
    All inputs and outputs are given in units of Angstroms.

    # Arguments:
    - sigma_i::Float64 : Width of primitive corresponding to Cartesian term i.
    - sigma_j::Float64 : Width of primitive corresponding to Cartesian term j.

    # Returns:
    - sigma_ij::Float64 : The computed σ_{ij} value.
    """

    return (sigma_i * sigma_j) / get_sigma_bar_ij(sigma_i, sigma_j)
end

@inline function get_r_ij(x_i::T, y_i::T, z_i::T, x_j::T, y_j::T, z_j::T)::T where {T<:AbstractFloat}
    """
    Compute the distance between two atoms.

        r_{ij} = √((x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2),

    where x_i and x_j are the x-coordinates of the atoms corresponding to Cartesian terms i and j, respectively,
    and similarly for y and z. All inputs and outputs are given in units of Angstroms.

    # Arguments:
    - x_i::T: x-coordinate of atom corresponding to Cartesian term i.
    - y_i::T: y-coordinate of atom corresponding to Cartesian term i.
    - z_i::T: z-coordinate of atom corresponding to Cartesian term i.
    - x_j::T: x-coordinate of atom corresponding to Cartesian term j.
    - y_j::T: y-coordinate of atom corresponding to Cartesian term j.
    - z_j::T: z-coordinate of atom corresponding to Cartesian term j.

    # Returns:
    - r_ij::T: The computed distance between atoms corresponding to Cartesian terms i and j.
    """

    return sqrt(
        (x_i - x_j) * (x_i - x_j) +
            (y_i - y_j) * (y_i - y_j) +
            (z_i - z_j) * (z_i - z_j)
    )
end

@inline function get_X_ij(x_i::T, x_j::T, sigma_i::T, sigma_j::T)::T where {T<:AbstractFloat}
    """
    Compute the Gaussian width weighted coordinate for the atoms corresponding to Cartesian terms i and j.
    This is defined by:

        X_{ij} = σ_{ij}^2 * (x_i / σ_i^2 + x_j / σ_j^2),

    where σ_i and σ_j are the widths corresponding to Cartesian terms i and j, respectively,
    and r_i and r_j are the position vectors of the atoms corresponding to Cartesian terms i and j, respectively.
    All inputs and outputs are given in units of Angstroms.

    # Arguments:
    - x_i::T: Coordinate of atom corresponding to Cartesian term i.
    - x_j::T: Coordinate of atom corresponding to Cartesian term j.
    - sigma_i::T: Width of primitive corresponding to Cartesian term i.
    - sigma_j::T: Width of primitive corresponding to Cartesian term j.

    # Returns:
    - X_ij::T: The computed X_{ij} coordinate.
    """

    typed_one = one(T)

    sigma_ij = get_sigma_ij(sigma_i, sigma_j)
    inv_sigma_i_sq = typed_one / (sigma_i * sigma_i) # Precompute for efficiency.
    inv_sigma_j_sq = typed_one / (sigma_j * sigma_j)
    return sigma_ij * sigma_ij * (x_i * inv_sigma_i_sq + x_j * inv_sigma_j_sq)
end

@inline function get_Mij(
        d_i::T, d_j::T, k_i::T, k_j::T,
        sigma_ij::T, sigma_bar_ij::T, r_ij::T
    )::T where {T<:AbstractFloat}
    """
    Compute the Cartesian pair weight between Cartesian terms i and j, defined as:

        M_{ij} = (d_i * d_j) * (k_i * k_j)* σ_{ij}^3 * exp(-r_{ij}^2 / (2 * σb_{ij}^2)),

    where d_i and d_j are the fully normalised primitive prefactors and k_i and k_j are the
    Cartesian prefactors for Cartesian terms i and j, respectively.

    # Arguments:
    - d_i::T: Normalised primitive prefactor for Cartesian term i.
    - d_j::T: Normalised primitive prefactor for Cartesian term j.
    - k_i::T: Cartesian prefactor for Cartesian term i.
    - k_j::T: Cartesian prefactor for Cartesian term j.
    - sigma_ij::T: The computed σ_{ij} value.
    - sigma_bar_ij::T: The computed σb_{ij} value.
    - r_ij::T: The computed distance between atoms corresponding to Cartesian terms i and j.

    # Returns:
    - M_ij::T: The computed Cartesian pair weight between Cartesian terms i and j.
    """

    return (d_i * d_j) * (k_i * k_j) * (sigma_ij * sigma_ij * sigma_ij) * exp(-T(0.5) * (r_ij * r_ij) / (sigma_bar_ij * sigma_bar_ij))
end

function construct_pair_coefficients(mol::MoleculeData{T})::Tuple{Array{T, 2}, Array{T, 2}, Array{T, 3}, Array{T, 2}, Array{T, 3}} where {T<:AbstractFloat}
    """
    Construct all pair coefficients, and return only the ones that will need to be used later in the
    form factor calculations. These are M_{ij}, σ_{ij}, and R_{ij}.

    # Arguments:
    - mol::MoleculeData: The MoleculeData structure containing all molecule and basis set information.

    # Returns:
    - M_ij::Array{T,2}: The Cartesian pair weights for all Cartesian term pairs.
    - sigma_ij::Array{T,2}: The σ_{ij} values for all Cartesian term pairs.
    - R_ij::Array{T,3}: The R_{ij} vectors for all Cartesian term pairs.
    - R_ij_mod:: Array{T,2}: The magnitudes of R_{ij} for all Cartesian term pairs.
    - R_ij_hat::Array{T,3}: The (θ, ϕ) angles of the unit vectors along R_{ij} for all Cartesian term pairs.
    """

    n_cartesian_terms = mol.n_cartesian_terms

    # Allocate memory for all coefficients to be returned.
    M_ij = Array{T}(undef, n_cartesian_terms, n_cartesian_terms)
    sigma_ij = Array{T}(undef, n_cartesian_terms, n_cartesian_terms)
    R_ij = Array{T}(undef, n_cartesian_terms, n_cartesian_terms, 3) # Last dimension is for x,y,z components.
    R_ij_mod = Array{T}(undef, n_cartesian_terms, n_cartesian_terms)
    R_ij_hat = Array{T}(undef, n_cartesian_terms, n_cartesian_terms, 2) # Last dimension is for (θ, ϕ) angles.

    @inbounds for i in 1:n_cartesian_terms
        atom_idx_i = mol.cartesian_term_to_atom[i]
        orbital_idx_i = mol.cartesian_term_to_orbital[i]
        primitive_idx_i = mol.cartesian_term_to_primitive[i]

        # Extract required quantities for Cartesian term i.
        x_i = mol.atom_coordinates[atom_idx_i, 1]
        y_i = mol.atom_coordinates[atom_idx_i, 2]
        z_i = mol.atom_coordinates[atom_idx_i, 3]
        sigma_i = mol.widths[primitive_idx_i]
        d_i = mol.normalised_coefficients[primitive_idx_i]
        k_i = mol.cartesian_prefactor[i]

        # Start from i and exploit the symmetry of all objects, with the exception of TDM_ij.
        @inbounds for j in i:n_cartesian_terms
            atom_idx_j = mol.cartesian_term_to_atom[j]
            orbital_idx_j = mol.cartesian_term_to_orbital[j]
            primitive_idx_j = mol.cartesian_term_to_primitive[j]

            # Extract required quantities for Cartesian term j.
            x_j = mol.atom_coordinates[atom_idx_j, 1]
            y_j = mol.atom_coordinates[atom_idx_j, 2]
            z_j = mol.atom_coordinates[atom_idx_j, 3]
            sigma_j = mol.widths[primitive_idx_j]
            d_j = mol.normalised_coefficients[primitive_idx_j]
            k_j = mol.cartesian_prefactor[j]

            # Compute intermediate quantities.
            sigma_ij_val = get_sigma_ij(sigma_i, sigma_j)
            sigma_bar_ij_val = get_sigma_bar_ij(sigma_i, sigma_j)
            r_ij_val = get_r_ij(x_i, y_i, z_i, x_j, y_j, z_j)

            # Compute and store the required pair coefficients, and use symmetry where possible.
            M_ij[i, j] = get_Mij(d_i, d_j, k_i, k_j, sigma_ij_val, sigma_bar_ij_val, r_ij_val)
            M_ij[j, i] = get_Mij(d_j, d_i, k_j, k_i, sigma_ij_val, sigma_bar_ij_val, r_ij_val)
            sigma_ij[i, j] = sigma_ij_val
            sigma_ij[j, i] = sigma_ij_val

            # Compute the R_ij vector components. Do them one at a time to avoid allocations.
            R_ij_x = get_X_ij(x_i, x_j, sigma_i, sigma_j)
            R_ij_y = get_X_ij(y_i, y_j, sigma_i, sigma_j)
            R_ij_z = get_X_ij(z_i, z_j, sigma_i, sigma_j)

            R_ij[i, j, 1] = R_ij_x
            R_ij[i, j, 2] = R_ij_y
            R_ij[i, j, 3] = R_ij_z

            R_ij[j, i, 1] = R_ij_x # Symmetry.
            R_ij[j, i, 2] = R_ij_y
            R_ij[j, i, 3] = R_ij_z

            # Compute the unit vector angles (θ, ϕ) for R_ij.
            R_ij_mag = sqrt(R_ij_x * R_ij_x + R_ij_y * R_ij_y + R_ij_z * R_ij_z)

            # Handle zero or very small magnitude to avoid division by zero.
            if R_ij_mag < eps(T) * T(100)  # Use a threshold relative to the smallest representable number.
                theta_ij = zero(T)
                phi_ij = zero(T)
            else
                # Clamp the argument to acos to [-1, 1] to avoid numerical errors.
                cos_theta = clamp(R_ij_z / R_ij_mag, -one(T), one(T))
                theta_ij = acos(cos_theta)
                phi_ij = atan(R_ij_y, R_ij_x)
            end

            R_ij_hat[i, j, 1] = theta_ij
            R_ij_hat[i, j, 2] = phi_ij
            R_ij_mod[i, j] = R_ij_mag

            R_ij_hat[j, i, 1] = theta_ij # Symmetry.
            R_ij_hat[j, i, 2] = phi_ij
            R_ij_mod[j, i] = R_ij_mag
        end
    end

    return M_ij, sigma_ij, R_ij, R_ij_mod, R_ij_hat
end
end
