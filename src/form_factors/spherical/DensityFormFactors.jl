# This module constructs the extra spherical form factors needed by the diagonal crystal correction.

module DensityFormFactors

using StaticArrays
using SphericalHarmonics
using LinearAlgebra: norm

using ..ConstructRTensor: fill_spherical_bessel_column!, KEV_TO_INV_ANGSTROM
using ...FastPowers: fast_i_pow

export construct_nuclear_R, split_density_R_tensors

function construct_nuclear_R(
        atom_coordinates::Matrix{T},
        nuclear_charges::Vector{Int},
        q_grid::Vector{T},
        l_max::Int,
    )::Array{Complex{T}, 3} where {T<:AbstractFloat}
    """
    Construct the nuclear form factor coefficients

        Z_lm(q) = 4π i^l Σ_I Z_I j_l(q R_I) conj(Y_l^m(R̂_I)).

    # Arguments:
    - atom_coordinates::Matrix{T}: Molecular-frame nuclear positions, in Angstroms.
    - nuclear_charges::Vector{Int}: Z_I for each atom.
    - q_grid::Vector{T}: The q grid, in keV.
    - l_max::Int: The largest angular mode.

    # Returns:
    - Array{Complex{T},3}: Z_lm(q), with dimensions (1, n_q, n_keys).
    """

    # Get the dimensions.
    n_atoms = size(atom_coordinates, 1)
    length(nuclear_charges) == n_atoms || error(
        "There are $(n_atoms) atomic coordinates but $(length(nuclear_charges)) nuclear charges.")

    n_q = length(q_grid)
    n_keys = (l_max + 1)^2
    nuclear_R = zeros(Complex{T}, 1, n_q, n_keys)
    q_grid_invA = T(KEV_TO_INV_ANGSTROM) .* q_grid

    # Allocate buffers for the Bessel functions, globally, and at each q, as well as the spherical harmonics.
    bessel = Matrix{T}(undef, l_max + 1, n_q)
    bessel_buffer = Vector{Float64}(undef, l_max + 1)
    Ylm_cache = SphericalHarmonics.cache(l_max, SphericalHarmonics.FullRange)

    for atom in 1:n_atoms
        position = SVector{3, T}(atom_coordinates[atom, :])
        distance = norm(position)
        charge = T(nuclear_charges[atom])

        # At the molecular origin only j_0 survives, and Y_0^0 = 1/sqrt(4π).
        if distance <= eps(T)
            @inbounds for q_idx in 1:n_q
                nuclear_R[1, q_idx, 1] += charge * sqrt(T(4 * π))
            end
            continue
        end

        # Compute the spherical harmonics.
        theta = acos(clamp(position[3] / distance, -one(T), one(T)))
        phi = atan(position[2], position[1])
        computePlmcostheta!(Ylm_cache, theta, l_max)
        computeYlm!(Ylm_cache, theta, phi, l_max)
        Yvals = SphericalHarmonics.getY(Ylm_cache)

        # Compute the Bessel functions.
        @inbounds for q_idx in 1:n_q
            fill_spherical_bessel_column!(bessel, bessel_buffer, q_idx,
                                          q_grid_invA[q_idx] * distance, l_max)
        end

        # Construct the nuclear R tensor.
        @inbounds for l in 0:l_max
            angular_prefactor = T(4 * π) * fast_i_pow(l, T) * charge
            key_base = l * l + l + 1
            for m in -l:l
                angular = angular_prefactor * conj(Complex{T}(Yvals[(l, m)]))
                key = key_base + m
                @simd for q_idx in 1:n_q
                    nuclear_R[1, q_idx, key] += angular * bessel[l + 1, q_idx]
                end
            end
        end
    end

    return nuclear_R
end

function split_density_R_tensors(
        all_R::Array{Complex{T}, 3},
        n_transitions::Int,
        atom_coordinates::Matrix{T},
        nuclear_charges::Vector{Int},
        q_grid::Vector{T},
        l_max::Int;
        threshold::T = zero(T),
    ) where {T<:AbstractFloat}
    """
    Split the density batch, [R_s1, R_s2, ..., R_sn, R_g, dR_1,..., dR_n], where R_sn is
    the nth transition R tensor (f_lm), R_g is the ground state R tensor, for the charge density,
    denoted N_g in the notes, and dR_n difference R tensor, ΔN_n in the notes. Then build

        Ξ_lm(q) = N_lm^(g)(q) - Z_lm(q).

    The neutral q = 0 monopoles are checked before being set exactly to zero.

    # Arguments:
    - all_R::Array{Complex{T},3}: The complete batched R tensor.
    - n_transitions::Int: Number of transition and difference matrices in the batch.
    - atom_coordinates::Matrix{T}: Molecular-frame nuclear positions, in Angstroms.
    - nuclear_charges::Vector{Int}: Z_I for each atom.
    - q_grid::Vector{T}: The q grid, in keV.
    - l_max::Int: The largest angular mode.
    - threshold::T: The W-tensor threshold used to construct all_R.

    # Returns:
    - Tuple: The transition R tensor, difference R tensor and Xi R tensor.
    """

    # Check that we have the right number of R tensors to split up.
    expected = 2 * n_transitions + 1
    size(all_R, 1) == expected || error(
        "The diagonal-density batch has $(size(all_R, 1)) matrices, expected $(expected).")
    abs(q_grid[1]) <= eps(T) || error("The diagonal correction needs a q grid starting at zero.")

    # Split into transition, GS, and difference R tensors.
    transition_R = all_R[1:n_transitions, :, :]
    ground_R = all_R[(n_transitions + 1):(n_transitions + 1), :, :]
    difference_R = all_R[(n_transitions + 2):end, :, :]
    
    # Construct Ξ_lm(q).
    nuclear_R = construct_nuclear_R(atom_coordinates, nuclear_charges, q_grid, l_max)
    Xi_R = ground_R .- nuclear_R

    # Both objects are neutral. Validate before pinning the q = 0 values exactly, since even a tiny
    # numerical monopole would be amplified by the 1/G² Ewald kernel.
    charge_scale = max(T(sum(nuclear_charges)), one(T))
    tolerance = T(100) * max(eps(T), abs(threshold)) * charge_scale
    difference_residual = maximum(abs, @view difference_R[:, 1, 1])
    Xi_residual = abs(Xi_R[1, 1, 1])
    difference_residual <= tolerance || error(
        "The difference density is not neutral at q = 0: residual $(difference_residual), tolerance $(tolerance).")
    Xi_residual <= tolerance || error(
        "The ground-state charge form factor is not neutral at q = 0: residual $(Xi_residual), tolerance $(tolerance).")

    # The above checks that the origin is approximately neutral, so we don't lose anything. This now hard enforces it for stability.
    @views fill!(difference_R[:, 1, 1], zero(Complex{T}))
    Xi_R[1, 1, 1] = zero(Complex{T})

    return transition_R, difference_R, Xi_R
end

end
