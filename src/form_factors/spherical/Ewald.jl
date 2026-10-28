# This module implements the Ewald split of the intermolecular Coulomb coupling.

module Ewald

using StaticArrays
using SphericalHarmonics
using LinearAlgebra: norm, cross, dot, mul!, BLAS

using ..CrystalLattice: CrystalLatticeData, fold_to_bz, ALPHA_EM, HBAR_C_EV_ANGSTROM,
                        HARTREE_EV, BOHR_ANGSTROM
using ..DensityFormFactors: Delta_row
using ..ReadBasisSet: build_transition_pairs

export EwaldParameters, EwaldTruncation, D_TRUNCATION, J_TRUNCATION,
       choose_ewald_parameters, short_range_kernel, supercell_radius,
       EwaldLongRangeData, EwaldLongRangeBuffers, build_ewald_long_range, add_long_range!,
       compute_ewald_D, J_self_matrix, check_dipole_against_oscillator_strength

# Coefficients below this fraction of the largest one over the long-range q range are treated as
# absent, which sets the max ℓ in the long-range sum. With this set to 1e-4, the error is around
# 1e-10 eV, more or less zero. To be revisitied, and extended to the normal SCarFFF. 
const COEFFICIENT_THRESHOLD = 1.0e-4

# |Q| below this is taken as the Q = 0 term, which happens only at k = 0 with G = 0. Allows for
# rounding errors.
const Q_ZERO_TOLERANCE = 1.0e-10


struct EwaldTruncation
    """
    Stores the measured constants that determine the truncation error of the Ewald splits.

        δ_R ≃ A_R exp[-η²(R_max - R_0)²],      R_0 = k_R (V_cell/Z_mol)^{1/3},
        δ_G ≃ A_G exp[-G_max²/4η_eff²],        1/η_eff² = 1/η² + 1/σ_0².

    R_0 accounts for the fact that elements of charge can be closer than the molecular centres,
    whilst σ_0 accounts for the natural decay of the form factor with q.

    # Fields:
    - A_R::Float64: The real-space prefactor.
    - A_G::Float64: The reciprocal-space prefactor.
    - sigma_0::Float64: The form factor's own reciprocal decay width σ_0, in Å^{-1}, or infinite for
      a density whose transform does not decay.
    - k_R::Float64: The real-space offset R_0 in units of the volume per molecule, (V_cell/Z_mol)^{1/3}.
    """
    A_R::Float64
    A_G::Float64
    sigma_0::Float64
    k_R::Float64
end

# Fitted parameters for the truncation errors.
const D_TRUNCATION = EwaldTruncation(2.0, 15.0, Inf, 1.5)
const J_TRUNCATION = EwaldTruncation(0.01, 0.4, 2.5, 1.5)

# How many passes the work minimisation is allowed before giving up.
const WORK_PASSES = 50

# The biggest truncation error that can be requested. Anything below this is pinned to this value.
const MAX_EPSILON_TOT = 0.01

# The tightest truncation error the constants above were measured against, so warn if the user asks for smaller.
const TESTED_EPSILON_TOT = 1.0e-6

struct EwaldParameters{T<:AbstractFloat}
    """
    The Ewald splitting parameter and the two cutoffs it implies.

    The cutoffs are pinned to a common estimated error ε_tot/√2 on both ends, so that neither dominates:

        δ_R = δ_G = ε_tot/√2.

    G_max and R_max are then chosen such that the amount of work done is minimised.

    # Fields:
    - eta::T: The splitting parameter η, in Å^{-1}.
    - R_max::T: The real-space cutoff for the short-range sum, in Å.
    - G_max::T: The reciprocal-space cutoff for the long-range sum, in Å^{-1}. The sum runs over
      reciprocal lattice vectors G, but the term dropped at the cutoff is the one at Q = k + G, so
      this bounds |Q| rather than |G|. See add_long_range!.
    - epsilon_tot::T: The truncation error the cutoffs are pinned to.
    """
    eta::T
    R_max::T
    G_max::T
    epsilon_tot::T
end

function choose_ewald_parameters(
        lattice::CrystalLatticeData{T},
        Z_mol::Integer,
        epsilon_tot::Real,
        truncation::EwaldTruncation;
        eta::Real = 0,
        cost_ratio::Real = 1,
    )::EwaldParameters{T} where {T<:AbstractFloat}
    """
    Choose η and the two cutoffs of one coupling's (J or D) split for a requested total error, ε_tot.

    Pinning both truncation errors to ε_tot/√2 leaves one free parameter, which is fixed by
    minimising the total work

        W = c_R N_R + c_G N_G,     N_R ≃ 4π R_max³ / 3 V_cell,   N_G ≃ V_cell G_max³ / 6π².

    Taking G_max as the free one, the error on the reciprocal sum fixes η and the error on the real
    space sum then fixes R_max,

        1/η² = 4 L_G/G_max² - 1/σ_0²,    R_max = R_0 + √L_R / η,    L_{R/G} = ln(√2 A_{R/G}/ε_tot),

    and dW/dG_max = 0 leaves

        G_max⁵ = (32π³ L_R L_G / V_cell²) R_max²/(R_max - R_0) (3c_R + R_max c_R')/(3c_G + G_max c_G'),

    with c_R' = dc_R/dR_max. Constant costs make the last factor c_R/c_G, which is what we use.

    # Arguments:
    - lattice::CrystalLatticeData{T}: The crystal lattice, for the cell volume and reciprocal spacings.
    - Z_mol::Integer: The number of molecules in the unit cell, which sets the volume per molecule.
    - epsilon_tot::Real: The requested truncation error, in (0, 1).
    - truncation::EwaldTruncation: The convergence constants of the coupling being split.
    - eta::Real: A fixed η in Å^{-1}, or 0 to derive one.
    - cost_ratio::Real: c_G / c_R, the cost of one reciprocal lattice vector relative to one real one.

    # Returns:
    - EwaldParameters{T}: The splitting parameter and both cutoffs.
    """

    # Check that an error ∈ [0,1], a cost ratio > 0 and at least one molecule are given.
    zero(epsilon_tot) < epsilon_tot < one(epsilon_tot) ||
        error("The Ewald truncation error must lie in (0, 1), got $(epsilon_tot).")
    cost_ratio > 0 || error("The Ewald cost ratio must be positive, got $(cost_ratio).")
    Z_mol >= 1 || error("The unit cell must hold at least one molecule, got $(Z_mol).")
    min(truncation.A_R, truncation.A_G) > 0 || error(
        "The Ewald truncation prefactors must be positive, got A_R = $(truncation.A_R), " *
        "A_G = $(truncation.A_G).")

    # Asking for 50% was never meant literally, so tighten such a request rather than refuse it.
    epsilon = min(T(epsilon_tot), T(MAX_EPSILON_TOT))
    epsilon < T(epsilon_tot) && @warn(
        "The requested Ewald truncation error is too big, so it " *
        "has been tightened.", requested = epsilon_tot, used = epsilon)
    # The laws are fits, so a request tighter than they were measured over is an extrapolation.
    # Said once a session, since it is about the constants rather than about any one crystal.
    epsilon < T(TESTED_EPSILON_TOT) && @warn(
        "The requested Ewald truncation error is tighter than the truncation laws were fitted " *
        "to, so the true truncation error may differ significantly from what is being requested.",
        requested = epsilon, tested_down_to = TESTED_EPSILON_TOT, maxlog = 1)

    # Extra factor of √2 since total error is √(δ_G^2 + δ_R^2) = √2 δ.
    L_R = log(sqrt(T(2)) * T(truncation.A_R) / epsilon)
    L_G = log(sqrt(T(2)) * T(truncation.A_G) / epsilon)
    # Only reachable by changing the table above, since MAX_EPSILON_TOT sits inside every law there.
    min(L_R, L_G) > 0 || error(
        "A_R = $(truncation.A_R) and A_G = $(truncation.A_G) cannot express an error of " *
        "$(epsilon), so no cutoff is implied. Lower MAX_EPSILON_TOT below √2 times the smaller.")

    # The offset, R_0, and the form factor decay width, 1/σ_0^2.
    R_0 = T(truncation.k_R) * (lattice.V_cell / T(Z_mol))^(one(T) / 3)
    inverse_sigma_squared = inv(T(truncation.sigma_0)^2)
    inverse_cost = inv(T(cost_ratio))

    # Find the η and R_max that give the requested error.
    split_at(G) = begin
        inverse_eta_squared = 4 * L_G / G^2 - inverse_sigma_squared
        inverse_eta_squared > 0 || error(
            "The reciprocal cutoff $(G) Å^{-1} is wider than the form factor's own decay allows " *
            "(σ_0 = $(truncation.sigma_0) Å^{-1}), so no η delivers the requested error.")
        eta_value = inv(sqrt(inverse_eta_squared))
        return eta_value, R_0 + sqrt(L_R) / eta_value
    end

    if eta > 0
        eta_value = T(eta)
        # The same two errors, but with η taken as given rather than derived.
        eta_eff = inv(sqrt(inv(eta_value^2) + inverse_sigma_squared))
        return EwaldParameters{T}(eta_value, R_0 + sqrt(L_R) / eta_value,
                                  2 * eta_eff * sqrt(L_G), epsilon)
    end

    # Start with a simple estimate for G_max, based on R_0 = 0, then iterate to find the true solution.
    G_max = 2 * sqrt(T(π)) * (inverse_cost / lattice.V_cell^2)^(one(T) / 6) * (L_R * L_G)^(one(T) / 4)
    work_scale = 32 * T(π)^3 * inverse_cost * L_R * L_G / lattice.V_cell^2
    converged = false
    for _ in 1:WORK_PASSES
        previous = G_max
        _, R_max = split_at(G_max)
        G_max = (work_scale * R_max^2 / (R_max - R_0))^(one(T) / 5)
        if abs(G_max - previous) <= sqrt(eps(T)) * G_max
            converged = true
            break
        end
    end
    converged || error(
        "The Ewald work minimisation did not settle on a reciprocal cutoff in $(WORK_PASSES) passes.")

    # Take the η and R_max corresponding to G_max.
    eta_value, R_max = split_at(G_max)

    return EwaldParameters{T}(eta_value, R_max, G_max, epsilon)
end

function supercell_radius(lattice::CrystalLatticeData{T})::T where {T<:AbstractFloat}
    """
    The distance to the furthest corner of the 3x3x3 supercell, which is the largest |ΔR| among the
    lattice vectors with all three indices in {-1, 0, 1}. This does not necessarily include all molecules in the
    3x3x3 supercell in the short-range sum, and may include some outside. Reported alongside R_max,
    as a familiar length to compare the chosen cutoff against.

    # Arguments:
    - lattice::CrystalLatticeData{T}: The crystal lattice.

    # Returns:
    - T: The radius, in Å.
    """

    # Get the lattice vectors.
    a1 = SVector{3, T}(lattice.direct[1, :])
    a2 = SVector{3, T}(lattice.direct[2, :])
    a3 = SVector{3, T}(lattice.direct[3, :])

    radius = zero(T)
    # Find the largest lattice vector in the 3x3x3 supercell.
    for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
        radius = max(radius, norm(s1 * a1 + s2 * a2 + s3 * a3))
    end

    return radius
end

function short_range_kernel(q_grid_invA::Vector{T}, eta::T)::Vector{T} where {T<:AbstractFloat}
    """
    The short-range Ewald kernel, [1 - exp(-q²/4η²)].

    # Arguments:
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å.
    - eta::T: The splitting parameter η, in Å^{-1}.

    # Returns:
    - Vector{T}: The kernel on the q grid.
    """
    return [one(T) - exp(-(q / (2 * eta))^2) for q in q_grid_invA]
end

struct EwaldLongRangeData{T<:AbstractFloat}
    """
    Everything the long-range sum needs that does not depend on k.

    The form factor coefficients are held as a compact sub-table covering only q <= G_max and
    ℓ <= l_max_lr, alongside the Hermite tangents used to interpolate them. That table is small
    enough to stay in cache, which matters because it is read at every q point of the outer grid.

    # Fields:
    - lattice::CrystalLatticeData{T}: The crystal lattice, used to fold k before summing.
    - G_vectors::Vector{SVector{3, T}}: Candidate reciprocal lattice vectors, in Å^{-1}.
    - f_table::Array{Complex{T}, 3}: The image coefficients f, with dimensions (n_keys_lr, n_lambda_i, n_q_lr).
    - f_tangent::Array{Complex{T}, 3}: The Hermite tangents Δq df/dq, with the same dimensions.
    - q_step::T: The spacing of the (uniform) q grid, in Å^{-1}.
    - l_max_lr::Int: The ℓ ceiling of the sub-table.
    - tau::Vector{SVector{3, T}}: The distinct τ_{A,i}, one per molecule in the cell.
    - image_of::Vector{Int}: Which of those tau each λ_i carries, so its length is n_lambda_i.
    - dipole_slopes::Matrix{Complex{T}}: lim_{q→0} f_{1m}(q)/q for each λ_i, columns m = -1, 0, 1, in Å. Used for the boundary term.
    - J_self::Matrix{Complex{T}}: J^self_{λ_iλ_j′} for λ_i, λ_j′ on the same image, in eV.
    - eta::T: The splitting parameter η, in Å^{-1}.
    - G_max::T: The reciprocal-space cutoff, in Å^{-1}.
    - prefactor::T: 4π α_EM ħc / V_cell, in eV Å^{-2}.
    - epsilon_b::T: The permittivity of the medium around the sample. Not to be confused with the Ewald truncation
      error, which EwaldParameters calls epsilon_tot.
    """
    lattice::CrystalLatticeData{T}
    G_vectors::Vector{SVector{3, T}}
    f_table::Array{Complex{T}, 3}
    f_tangent::Array{Complex{T}, 3}
    q_step::T
    l_max_lr::Int
    tau::Vector{SVector{3, T}}
    image_of::Vector{Int}
    dipole_slopes::Matrix{Complex{T}}
    J_self::Matrix{Complex{T}}
    eta::T
    G_max::T
    prefactor::T
    epsilon_b::T
end

struct EwaldLongRangeBuffers{T<:AbstractFloat, C}
    """
    Per-task buffers for the long-range sum. One of these per BlochEigensystem, so it
    is allocated once per thread rather than once per q point.

    # Fields:
    - f_image::Vector{Complex{T}}: The phase-shifted molecular amplitudes f_{λ_i}(Q).
    - harmonics::Vector{Complex{T}}: Y_ℓ^μ(Q̂) flattened onto the table's key ordering.
    - image_phases::Vector{Complex{T}}: exp(i Q . τ) for each molecule in the cell.
    - Ylm_cache::C: The spherical harmonic cache, sized to the ℓ ceiling of the sub-table.
    """
    f_image::Vector{Complex{T}}
    harmonics::Vector{Complex{T}}
    image_phases::Vector{Complex{T}}
    Ylm_cache::C
end

function EwaldLongRangeBuffers(data::EwaldLongRangeData{T}) where {T<:AbstractFloat}
    cache = SphericalHarmonics.cache(data.l_max_lr, SphericalHarmonics.FullRange)
    return EwaldLongRangeBuffers{T, typeof(cache)}(
        Vector{Complex{T}}(undef, length(data.image_of)),
        Vector{Complex{T}}(undef, (data.l_max_lr + 1)^2),
        Vector{Complex{T}}(undef, length(data.tau)),
        cache,
    )
end

function reciprocal_plane_spacings(lattice::CrystalLatticeData{T})::NTuple{3, T} where {T<:AbstractFloat}
    """
    The spacing between planes of the reciprocal lattice along each axis, which is its cell volume
    divided by the area of the opposite face. See the analogous construct in the short range term for
    details.

    # Arguments:
    - lattice::CrystalLatticeData{T}: The crystal lattice.

    # Returns:
    - NTuple{3, T}: The three spacings, in Å^{-1}.
    """

    # Get the reciprocal lattice vectors and cell volume.
    b1 = SVector{3, T}(lattice.reciprocal[1, :])
    b2 = SVector{3, T}(lattice.reciprocal[2, :])
    b3 = SVector{3, T}(lattice.reciprocal[3, :])

    reciprocal_volume = abs(dot(b1, cross(b2, b3)))

    # Compute the interplanar spacings, which are the cell volume divided by the area of the opposite face 
    return (
        reciprocal_volume / norm(cross(b2, b3)),
        reciprocal_volume / norm(cross(b3, b1)),
        reciprocal_volume / norm(cross(b1, b2)),
    )
end

function enumerate_reciprocal_vectors(
        lattice::CrystalLatticeData{T},
        cutoff::T,
    )::Vector{SVector{3, T}} where {T<:AbstractFloat}
    """
    Enumerate the reciprocal lattice vectors G with |G| <= cutoff, including G = 0.

    # Arguments:
    - lattice::CrystalLatticeData{T}: The crystal lattice.
    - cutoff::T: The cutoff on |G|, in Å^{-1}.

    # Returns:
    - Vector{SVector{3, T}}: The reciprocal lattice vectors, in Å^{-1}.
    """

    # Get the reciprocal vectors and plane spacings.
    b1 = SVector{3, T}(lattice.reciprocal[1, :])
    b2 = SVector{3, T}(lattice.reciprocal[2, :])
    b3 = SVector{3, T}(lattice.reciprocal[3, :])

    spacings = reciprocal_plane_spacings(lattice)
    # Find the search bounds. This the number of plane spacings less than the cutoff, plus one for wiggle room.
    search = ntuple(i -> Int(ceil(cutoff / spacings[i])) + 1, 3)

    vectors = SVector{3, T}[]
    # Keep only the vectors in the search range less than the cutoff distance from the origin.
    for n1 in -search[1]:search[1], n2 in -search[2]:search[2], n3 in -search[3]:search[3]
        vector = n1 * b1 + n2 * b2 + n3 * b3
        norm(vector) <= cutoff && push!(vectors, vector)
    end

    return vectors
end

function brillouin_zone_radius(lattice::CrystalLatticeData{T})::T where {T<:AbstractFloat}
    """
    The largest |k| that fold_to_bz can return. It puts k in the parallelepiped with fractional
    coordinates in [-1/2, 1/2), so the extreme points are the corners, and the radius is half the
    longest body diagonal of the reciprocal cell. Used to determine the list of G vectors.

    # Arguments:
    - lattice::CrystalLatticeData{T}: The crystal lattice.

    # Returns:
    - T: The bounding radius of the fundamental domain, in Å^{-1}.
    """

    # Get the lattice vectors.
    b1 = SVector{3, T}(lattice.reciprocal[1, :])
    b2 = SVector{3, T}(lattice.reciprocal[2, :])
    b3 = SVector{3, T}(lattice.reciprocal[3, :])

    radius = zero(T)
    # Find the maximum radius.
    for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
        radius = max(radius, norm(s1 * b1 + s2 * b2 + s3 * b3) / 2)
    end

    return radius
end

function long_range_l_ceiling(
        rotated_R::Array{Complex{T}, 3},
        n_q_lr::Int,
        l_max::Int,
    )::Int where {T<:AbstractFloat}
    """
    Find the largest ℓ whose coefficients are non-negligible for the long-range q grid.

    # Arguments:
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients f, with dimensions (n_lambda_i, n_q, n_keys).
    - n_q_lr::Int: How many q points the long-range sum reaches.
    - l_max::Int: The maximum angular momentum mode of the form factor.

    # Returns:
    - Int: The ℓ ceiling.
    """

    n_lambda_i = size(rotated_R, 1)

    # Find the largest coefficient in each l block over the long-range q range.
    block_magnitude = zeros(T, l_max + 1)
    @inbounds for l in 0:l_max, key in (l * l + 1):((l + 1)^2), q_idx in 1:n_q_lr, lambda_i in 1:n_lambda_i
        block_magnitude[l + 1] = max(block_magnitude[l + 1], abs(rotated_R[lambda_i, q_idx, key]))
    end

    cutoff = T(COEFFICIENT_THRESHOLD) * maximum(block_magnitude)
    # Work backwards to find the first (largest ℓ) block with a magnitude greater than the cutoff. Keep everything below.
    for l in l_max:-1:0
        block_magnitude[l + 1] > cutoff && return l
    end

    return 0
end

function build_coefficient_table(
        rotated_R::Array{Complex{T}, 3},
        q_grid_invA::Vector{T},
        l_max::Int,
        G_max::T,
    ) where {T<:AbstractFloat}
    """
    Build the compact coefficient and Hermite-tangent tables used by a reciprocal sum.

    # Arguments:
    - rotated_R::Array{Complex{T},3}: Coefficients for every object on the full q grid.
    - q_grid_invA::Vector{T}: Uniform q grid starting at zero, in inverse Angstroms.
    - l_max::Int: The largest angular mode in rotated_R.
    - G_max::T: Reciprocal-space cutoff, in inverse Angstroms.

    # Returns:
    - Tuple: Compact coefficients, Hermite tangents, q spacing and retained l ceiling.
    """

    # Get the dimensions and grid spacing.
    n_objects = size(rotated_R, 1)
    n_q = length(q_grid_invA)
    n_q >= 3 || error("The Ewald interpolation needs at least three q points, got $(n_q).")
    
    # Check the q grid is uniform and starts at zero.
    q_step = q_grid_invA[2] - q_grid_invA[1]
    abs(q_grid_invA[1]) < eps(T) || error("The Ewald q grid must start at zero.")
    # Use sqrt(eps) here, to make sure it is compatible with float32. This will still be well below
    # any meaningul deviations.
    uniform_tolerance = sqrt(eps(T)) * q_step
    all(abs(q_grid_invA[i + 1] - q_grid_invA[i] - q_step) < uniform_tolerance
        for i in 1:(n_q - 1)) || error("The Ewald q grid must be uniform.")

    # Check that we can actually reach G_max with the grid and error settings.
    q_grid_invA[end] >= G_max || error(
        "The Ewald cutoff $(G_max) Å^-1 lies beyond the q grid at $(q_grid_invA[end]) Å^-1.")

    # Trim to the q grid needed up to G_max, ensuring we have the extra points for the interpolation.
    n_q_lr = min(n_q, Int(ceil(G_max / q_step)) + 3)
    # Truncate l_max as well.
    l_max_lr = long_range_l_ceiling(rotated_R, n_q_lr, l_max)
    n_keys_lr = (l_max_lr + 1)^2

    # Store the subset of rotated coefficients that we need for the LR sum.
    table = Array{Complex{T}, 3}(undef, n_keys_lr, n_objects, n_q_lr)
    @inbounds for q_idx in 1:n_q_lr, object in 1:n_objects, key in 1:n_keys_lr
        table[key, object, q_idx] = rotated_R[object, q_idx, key]
    end

    # Also store the tangents for the Hermite interpolation. Premultiply by dx (i.e. don't divide by it) to save multiplying later.
    # The cubic spline is accurate to dx^4, so a use a five point derivative stencil to match.
    tangent = similar(table)
    @inbounds for object in 1:n_objects, key in 1:n_keys_lr
        for q_idx in 3:(n_q_lr - 2)
            tangent[key, object, q_idx] =
                (-table[key, object, q_idx + 2] + 8 * table[key, object, q_idx + 1]
                 - 8 * table[key, object, q_idx - 1] + table[key, object, q_idx - 2]) / 12
        end
        if n_q_lr >= 5
            # The two nodes at each end have no central stencil, so use the one-sided five point
            # ones instead of a plain difference.
            tangent[key, object, 1] =
                (-25 * table[key, object, 1] + 48 * table[key, object, 2]
                 - 36 * table[key, object, 3] + 16 * table[key, object, 4]
                 - 3 * table[key, object, 5]) / 12
            tangent[key, object, 2] =
                (-3 * table[key, object, 1] - 10 * table[key, object, 2]
                 + 18 * table[key, object, 3] - 6 * table[key, object, 4]
                 + table[key, object, 5]) / 12
            tangent[key, object, n_q_lr - 1] =
                (3 * table[key, object, n_q_lr] + 10 * table[key, object, n_q_lr - 1]
                 - 18 * table[key, object, n_q_lr - 2] + 6 * table[key, object, n_q_lr - 3]
                 - table[key, object, n_q_lr - 4]) / 12
            tangent[key, object, n_q_lr] =
                (25 * table[key, object, n_q_lr] - 48 * table[key, object, n_q_lr - 1]
                 + 36 * table[key, object, n_q_lr - 2] - 16 * table[key, object, n_q_lr - 3]
                 + 3 * table[key, object, n_q_lr - 4]) / 12
        else
            # Fewer than five nodes, so there is nothing to build the one-sided stencils from.
            tangent[key, object, 1] = table[key, object, 2] - table[key, object, 1]
            tangent[key, object, 2] =
                (table[key, object, 3] - table[key, object, 1]) / 2
            tangent[key, object, n_q_lr - 1] =
                (table[key, object, n_q_lr] - table[key, object, n_q_lr - 2]) / 2
            tangent[key, object, n_q_lr] =
                table[key, object, n_q_lr] - table[key, object, n_q_lr - 1]
        end
    end

    return table, tangent, q_step, l_max_lr
end

function build_ewald_long_range(
        rotated_R::Array{Complex{T}, 3},
        rotated_R_bc::Array{Complex{T}, 3},
        tau_of_lambda_i::Vector{SVector{3, T}},
        image_of::Vector{Int},
        lattice::CrystalLatticeData{T},
        q_grid_invA::Vector{T},
        q_bc_invA::T,
        l_max::Int,
        parameters::EwaldParameters{T};
        epsilon_b::Real = 1,
    )::EwaldLongRangeData{T} where {T<:AbstractFloat}
    """
    Assemble everything the long-range sum

        J_bar^LR_{λ_iλ_j′}(k) = (4π α_EM ħc / V_cell) Σ_{Q ≠ 0}^{G_max} (1/Q²) exp(-Q²/4η²) f_{λ_i}(Q) conj(f_{λ_j′}(Q)),
        
    with
        
        f_{λ_i}(Q) = exp(i Q . τ_{A,i}) Σ_{ℓm} f^{λ_i}_{ℓm}(Q) Y_ℓ^μ(Q̂),
        
    and Q = k + G,

    needs that does not depend on k: the G vectors, the interpolation table for f, the
    transition dipole slopes for the Q = 0 term, and the self-interaction that has to be subtracted.

    # Arguments:
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients f, with dimensions (n_lambda_i, n_q, n_keys).
    - rotated_R_bc::Array{Complex{T}, 3}: The same coefficients at q_bc alone, which is where the
      transition dipole of the Q = 0 term is read off.
    - tau_of_lambda_i::Vector{SVector{3, T}}: The translation τ_{A,i} for each λ_i, in Å. Only the
      distinct values are stored, since the phase depends on the molecule rather than on λ_i.
    - image_of::Vector{Int}: The image index for each λ_i, which fixes which pairs share a molecule.
    - lattice::CrystalLatticeData{T}: The crystal lattice.
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å. Must be uniform and start at zero.
    - q_bc_invA::T: The |q| at which the transition dipole is read off, in inverse Å.
    - l_max::Int: The maximum angular momentum mode of the form factor.
    - parameters::EwaldParameters{T}: The splitting parameter and cutoffs.
    - epsilon_b::Real: The permittivity of the medium around the sample.

    # Returns:
    - EwaldLongRangeData{T}: The k-independent long-range data.
    """

    # Get the dimensions.
    n_lambda_i = size(rotated_R, 1)
    n_q = length(q_grid_invA)

    # A permittivity below 1 is unphysical, and would make the boundary term larger than the vacuum one.
    epsilon_b >= 1 || error("The medium permittivity must be at least 1, got $(epsilon_b).")

    # Check that we have enough points for interpolation.
    n_q >= 3 || error("The Ewald long-range sum needs at least three q points, got $(n_q).")
    # The dipole coefficients cover the same λ_i, but only the one q point they were evaluated at.
    size(rotated_R_bc, 1) == n_lambda_i && size(rotated_R_bc, 3) == size(rotated_R, 3) || error(
        "The dipole coefficients have size $(size(rotated_R_bc)), expected $((n_lambda_i, 1, size(rotated_R, 3))).")
    q_step = q_grid_invA[2] - q_grid_invA[1]
    # The Hermite interpolation assumes a uniform grid anchored at zero.
    abs(q_grid_invA[1]) < eps(T) ||
        error("The Ewald long-range sum expects a q grid starting at zero, got $(q_grid_invA[1]).")
    # Use sqrt(eps) here, to make sure it is compatible with float32. This will still be well below
    # any meaningul deviations.
    uniform_tolerance = sqrt(eps(T)) * q_step
    all(abs(q_grid_invA[i + 1] - q_grid_invA[i] - q_step) < uniform_tolerance for i in 1:(n_q - 1)) ||
        error("The Ewald long-range sum expects a uniform q grid.")

    # Check that we can actually reach G_max with the grid and error settings.
    q_grid_invA[end] >= parameters.G_max || error(
        "The Ewald reciprocal cutoff G_max = $(parameters.G_max) Å^{-1} lies beyond the q grid, " *
        "which reaches $(q_grid_invA[end]) Å^{-1}. Raise q_max or allow for a larger truncation error."
    )

    # Trim to the q grid needed up to G_max, ensuring we have the extra points for the interpolation.
    n_q_lr = min(n_q, Int(ceil(parameters.G_max / q_step)) + 3)
    # Truncate l_max as well.
    l_max_lr = long_range_l_ceiling(rotated_R, n_q_lr, l_max)
    n_keys_lr = (l_max_lr + 1)^2

    # Store the subset of rotated coefficients that we need for the LR sum.
    f_table = Array{Complex{T}, 3}(undef, n_keys_lr, n_lambda_i, n_q_lr)
    @inbounds for q_idx in 1:n_q_lr, lambda_i in 1:n_lambda_i, key in 1:n_keys_lr
        f_table[key, lambda_i, q_idx] = rotated_R[lambda_i, q_idx, key]
    end

    # Also store the tangents for the Hermite interpolation. Premultiply by dx (i.e. don't divide by it) to save multiplying later.
    # Endpoints use one directional difference, interior points use central difference, hence the 1/2 for them.
    f_tangent = Array{Complex{T}, 3}(undef, n_keys_lr, n_lambda_i, n_q_lr)
    @inbounds for lambda_i in 1:n_lambda_i, key in 1:n_keys_lr
        f_tangent[key, lambda_i, 1] = f_table[key, lambda_i, 2] - f_table[key, lambda_i, 1]
        for q_idx in 2:(n_q_lr - 1)
            f_tangent[key, lambda_i, q_idx] =
                (f_table[key, lambda_i, q_idx + 1] - f_table[key, lambda_i, q_idx - 1]) / 2
        end
        f_tangent[key, lambda_i, n_q_lr] = f_table[key, lambda_i, n_q_lr] - f_table[key, lambda_i, n_q_lr - 1]
    end

    # Compute the G vectors. The cutoff bounds |Q| = |k + G|, so a G as far out as G_max plus the
    # largest |k| in the zone can still be inside it at some k, and has to be on the list.
    G_vectors = enumerate_reciprocal_vectors(lattice, parameters.G_max + brillouin_zone_radius(lattice))

    # Compute the q -> 0 term, and self-interaction term.
    dipole_slopes = dipole_slope_matrix(rotated_R_bc, q_bc_invA, l_max)
    J_self = J_self_matrix(rotated_R, image_of, q_grid_invA, l_max, parameters.eta)

    prefactor = T(4 * π * ALPHA_EM * HBAR_C_EV_ANGSTROM) / lattice.V_cell

    # Store the tau for each image.
    n_images = maximum(image_of)
    tau = [tau_of_lambda_i[findfirst(==(i), image_of)] for i in 1:n_images]

    return EwaldLongRangeData{T}(
        lattice, G_vectors, f_table, f_tangent, q_step, l_max_lr,
        tau, copy(image_of),
        dipole_slopes, J_self, parameters.eta, parameters.G_max, prefactor, T(epsilon_b),
    )
end

function dipole_slope_matrix(
        rotated_R_bc::Array{Complex{T}, 3},
        q_bc_invA::T,
        l_max::Int,
    )::Matrix{Complex{T}} where {T<:AbstractFloat}
    """
    Read lim_{q→0} f_{1m}(q)/q for every λ_i and m off a single evaluation at q_bc.

    # Arguments:
    - rotated_R_bc::Array{Complex{T}, 3}: The rotated coefficients f at q_bc alone, with
      dimensions (n_lambda_i, 1, n_keys).
    - q_bc_invA::T: The |q| at which the dipole is read off, in inverse Å.
    - l_max::Int: The maximum angular momentum mode of the form factor.

    # Returns:
    - Matrix{Complex{T}}: The slopes, with dimensions (n_lambda_i, 3) for m = -1, 0, 1, in Å.
    """

    l_max >= 1 || error("The Ewald Q = 0 term needs the ℓ = 1 coefficients, but l_max = $(l_max).")
    size(rotated_R_bc, 2) == 1 || error(
        "The dipole coefficients should sit on a single q point, got $(size(rotated_R_bc, 2)).")
    q_bc_invA > 0 || error("The dipole evaluation point must be positive, got $(q_bc_invA).")

    n_lambda_i = size(rotated_R_bc, 1)
    slopes = Matrix{Complex{T}}(undef, n_lambda_i, 3)

    @inbounds for (column, m) in enumerate(-1:1)
        key = 1 * 1 + (1 + m) + 1
        for lambda_i in 1:n_lambda_i
            slopes[lambda_i, column] = rotated_R_bc[lambda_i, 1, key] / q_bc_invA
        end
    end

    return slopes
end

function check_dipole_against_oscillator_strength(
        R_bc::Array{Complex{T}, 3},
        q_bc_invA::T,
        Delta_E_eV::Vector{T},
        f_osc::Vector{T},
        l_max::Int;
        tolerance::T = T(1e-2),
        f_osc_floor::T = T(1e-3),
    )::T where {T<:AbstractFloat}
    """
    Check the dipole read off f_{1m}(q_bc)/q_bc against the oscillator strength from TD-DFT.

    TD-DFT gives f_osc = (2/3) ΔE |μ|² in atomic units without reference to the form factor.
    The ℓ = 1 coefficients give the same number through|μ|² = (3/4π) Σ_m |f_{1m}(q_bc)/q_bc|²,
    so the two disagreeing means the ℓ = 1 coefficients are wrong.

    # Arguments:
    - R_bc::Array{Complex{T}, 3}: Coefficients at q_bc alone, with dimensions (n_transitions, 1, n_keys).
    - q_bc_invA::T: The |q| at which the dipole is read off, in inverse Å.
    - Delta_E_eV::Vector{T}: The transition energies ΔE, in eV.
    - f_osc::Vector{T}: The oscillator strength of each transition, dimensionless.
    - l_max::Int: The maximum angular momentum mode of the form factor.
    - tolerance::T: The largest fractional disagreement allowed on any checked transition.
    - f_osc_floor::T: Transitions with an oscillator strength below this are not checked.

    # Returns:
    - T: The worst fractional disagreement found, or zero if no transition was strong enough.
    """

    # Get the dipole moments from the ℓ = 1 coefficients.
    slopes = dipole_slope_matrix(R_bc, q_bc_invA, l_max)
    n_transitions = size(R_bc, 1)
    length(Delta_E_eV) == n_transitions || error(
        "There are $(length(Delta_E_eV)) energies for $(n_transitions) transitions.")
    length(f_osc) == n_transitions || error(
        "There are $(length(f_osc)) oscillator strengths for $(n_transitions) transitions.")

    worst_error = zero(T)
    worst_transition = 0
    for transition in 1:n_transitions
        f_osc[transition] >= f_osc_floor || continue
        # Sum the three m components.
        total = sum(abs2(slopes[transition, column]) for column in 1:3)
        dipole_from_form_factor = sqrt(T(3 / (4 * π)) * total)
        # |μ|² = (3/2) f_osc / ΔE in atomic units, converted from Bohr to Å.
        energy_hartree = Delta_E_eV[transition] / T(HARTREE_EV)
        dipole_from_f_osc = sqrt(T(1.5) * f_osc[transition] / energy_hartree) * T(BOHR_ANGSTROM)
        relative_error = abs(dipole_from_form_factor - dipole_from_f_osc) / dipole_from_f_osc
        if relative_error > worst_error
            worst_error = relative_error
            worst_transition = transition
        end
    end

    worst_error <= tolerance || error(
        "The ℓ = 1 coefficients disagree with the oscillator strength by $(round(worst_error * 100, digits = 3))% " *
        "on transition $(worst_transition), which is more than the $(round(tolerance * 100, digits = 3))% allowed. " *
        "Either q_bc = $(q_bc_invA) Å^-1 is too large, or the ℓ = 1 coefficients are wrong."
    )

    return worst_error
end

function J_self_matrix(
        rotated_R::Array{Complex{T}, 3},
        image_of::Vector{Int},
        q_grid_invA::Vector{T},
        l_max::Int,
        eta::T,
    )::Matrix{Complex{T}} where {T<:AbstractFloat}
    """
    The self-interaction J^self_{λ_iλ_j′} that the reciprocal sum includes but the physics does not,
    since a molecule does not interact with itself. It is subtracted from the ΔR = 0 couplings.

    At zero separation the phase drops out and the full integral reduces to a single radial integral:

        J^self_{λ_iλ_j′} = (α_EM ħc / 2π²) ∫dq exp(-q²/4η²) Σ_{ℓm} f^{λ_i}_{ℓm}(q) conj(f^{λ_j′}_{ℓm}(q)).

    Only pairs on the same molecule are affected, so the matrix is zero between different images.

    # Arguments:
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients f, with dimensions (n_lambda_i, n_q, n_keys).
    - image_of::Vector{Int}: The image index for each λ_i.
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å.
    - l_max::Int: The maximum angular momentum mode of the form factor.
    - eta::T: The splitting parameter η, in Å^{-1}.

    # Returns:
    - Matrix{Complex{T}}: The self-interaction in eV, with dimensions (n_lambda_i, n_lambda_i).
    """

    # Get the dimensions.
    n_lambda_i = size(rotated_R, 1)
    n_q = length(q_grid_invA)
    n_keys = (l_max + 1)^2

    # The Gaussian kernel dies long before the q grid ends, so the trapezoidal rule is ample here.
    weights = Vector{T}(undef, n_q)
    @inbounds for q_idx in 1:n_q
        # Endpoints use one directional difference, interior use central difference.
        step = q_idx == 1 ? q_grid_invA[2] - q_grid_invA[1] :
               q_idx == n_q ? q_grid_invA[n_q] - q_grid_invA[n_q - 1] :
               (q_grid_invA[q_idx + 1] - q_grid_invA[q_idx - 1]) / 2
        # Endpoint and interior weights are different from interior.
        weights[q_idx] = (q_idx == 1 || q_idx == n_q ? step / 2 : step) *
                         exp(-(q_grid_invA[q_idx] / (2 * eta))^2)
    end

    prefactor = Complex{T}(ALPHA_EM * HBAR_C_EV_ANGSTROM / (2 * π^2))
    J_self = zeros(Complex{T}, n_lambda_i, n_lambda_i)

    @inbounds for lambda_j in 1:n_lambda_i, lambda_i in 1:lambda_j
        # Skip pairs on different molecules.
        image_of[lambda_i] == image_of[lambda_j] || continue

        total = zero(Complex{T})
        # Perform the integral.
        for key in 1:n_keys, q_idx in 1:n_q
            total += weights[q_idx] * rotated_R[lambda_i, q_idx, key] * conj(rotated_R[lambda_j, q_idx, key])
        end

        value = prefactor * total
        J_self[lambda_i, lambda_j] = value
        J_self[lambda_j, lambda_i] = conj(value)
    end

    return J_self
end

function compute_ewald_D(
        rotated_Delta::Array{Complex{T}, 3},
        rotated_Xi::Array{Complex{T}, 3},
        rotated_Delta_bc::Array{Complex{T}, 3},
        rotated_Xi_bc::Array{Complex{T}, 3},
        q_grid_invA::Vector{T},
        q_bc_invA::T,
        n_transitions::Int,
        l_max::Int,
        data::EwaldLongRangeData{T},
        parameters::EwaldParameters{T},
    )::Array{T, 3} where {T<:AbstractFloat}
    """
    Compute the long-range part of D

        D_LR = D_reciprocal + D_boundary - D_self.

    Xi is summed over the unit cell once for each G before it is contracted with every Δ. D only
    couples excitations on the same molecule, so what comes back is one block per image.

    # Arguments:
    - rotated_Delta::Array{Complex{T},3}: Δ, one row per image and pair of states.
    - rotated_Xi::Array{Complex{T},3}: Xi for every molecular image.
    - rotated_Delta_bc::Array{Complex{T},3}: Δ at q_bc alone, for the boundary term.
    - rotated_Xi_bc::Array{Complex{T},3}: Xi at q_bc alone, for the boundary term.
    - q_grid_invA::Vector{T}: Uniform q grid starting at zero, in inverse Angstroms.
    - q_bc_invA::T: The |q| at which the boundary term's dipoles are read off, in inverse Angstroms.
    - n_transitions::Int: The number of transitions per image, which fixes the block size.
    - l_max::Int: The largest angular mode.
    - data::EwaldLongRangeData{T}: The cell, the image positions and the boundary permittivity.
    - parameters::EwaldParameters{T}: D's own Ewald split.

    # Returns:
    - Array{T,3}: The real long-range D, with dimensions (n_images, n_transitions, n_transitions), in eV.
    """

    # Get the dimensions, and check the tensors match them.
    n_lambda_i = length(data.image_of)
    n_images = length(data.tau)
    n_keys = (l_max + 1)^2
    n_lambda_i == n_images * n_transitions || error(
        "There are $(n_lambda_i) excitations for $(n_images) images of $(n_transitions) transitions.")
    pairs = build_transition_pairs(n_transitions)
    n_pairs = length(pairs.transitions_of_pair)
    n_Delta_rows = n_images * n_pairs
    size(rotated_Delta, 1) == n_Delta_rows || error(
        "There are $(size(rotated_Delta, 1)) Δ densities for $(n_Delta_rows) image and pair combinations.")
    size(rotated_Delta, 2) == length(q_grid_invA) && size(rotated_Delta, 3) == n_keys || error(
        "The Δ tensor has size $(size(rotated_Delta)), expected $((n_Delta_rows, length(q_grid_invA), n_keys)).")
    size(rotated_Xi, 1) == n_images || error(
        "There are $(size(rotated_Xi, 1)) Ξ tensors for $(n_images) images.")
    size(rotated_Xi, 2) == length(q_grid_invA) && size(rotated_Xi, 3) == n_keys || error(
        "The Ξ tensor has size $(size(rotated_Xi)), expected $((n_images, length(q_grid_invA), n_keys)).")
    # The boundary tensors cover the same rows and images, but only the one q point they sit on.
    size(rotated_Delta_bc) == (n_Delta_rows, 1, n_keys) || error(
        "The boundary Δ tensor has size $(size(rotated_Delta_bc)), expected $((n_Delta_rows, 1, n_keys)).")
    size(rotated_Xi_bc) == (n_images, 1, n_keys) || error(
        "The boundary Ξ tensor has size $(size(rotated_Xi_bc)), expected $((n_images, 1, n_keys)).")

    # Build the compact tables for both densities. They share a q grid, so their spacings must agree.
    Delta_table, Delta_tangent, q_step, Delta_l_max =
        build_coefficient_table(rotated_Delta, q_grid_invA, l_max, parameters.G_max)
    Xi_table, Xi_tangent, Xi_q_step, Xi_l_max =
        build_coefficient_table(rotated_Xi, q_grid_invA, l_max, parameters.G_max)
    abs(q_step - Xi_q_step) <= eps(T) * max(q_step, one(T)) || error(
        "The Δ and Ξ interpolation grids do not match.")

    # The two densities may keep different l ceilings, so the harmonics must cover the larger.
    l_ceiling = max(Delta_l_max, Xi_l_max)
    harmonics = Vector{Complex{T}}(undef, (l_ceiling + 1)^2)
    Ylm_cache = SphericalHarmonics.cache(l_ceiling, SphericalHarmonics.FullRange)
    D_blocks = zeros(Complex{T}, n_images, n_transitions, n_transitions)

    @inbounds for G in enumerate_reciprocal_vectors(data.lattice, parameters.G_max)
        # Here k = 0, so Q = G. Skip G = 0, which is handled by the boundary term.
        G_norm < T(Q_ZERO_TOLERANCE) && continue

        # Compute the spherical harmonics for this G.
        theta = acos(clamp(G[3] / G_norm, -one(T), one(T)))
        phi = atan(G[2], G[1])
        computePlmcostheta!(Ylm_cache, theta, l_ceiling)
        computeYlm!(Ylm_cache, theta, phi, l_ceiling)
        Yvals = SphericalHarmonics.getY(Ylm_cache)

        # Flatten the harmonics.
        for l in 0:l_ceiling
            key_base = l * l + l + 1
            for m in -l:l
                harmonics[key_base + m] = Complex{T}(Yvals[(l, m)])
            end
        end

        # Cubic Hermite stencil for |G| on the uniform q grid.
        position = G_norm / q_step
        Delta_node = min(Int(floor(position)) + 1, size(Delta_table, 3) - 1)
        Xi_node = min(Int(floor(position)) + 1, size(Xi_table, 3) - 1)
        s = position - (Delta_node - 1)
        s2 = s * s
        s3 = s2 * s
        h00 = 2 * s3 - 3 * s2 + one(T)
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2

        # Sum the Ξ over all C_k once for this G, then use it for every Δ pair.
        total_Xi = zero(Complex{T})
        for image in 1:n_images
            amplitude = zero(Complex{T})
            @simd for key in 1:size(Xi_table, 1)
                amplitude += (h00 * Xi_table[key, image, Xi_node] +
                              h10 * Xi_tangent[key, image, Xi_node] +
                              h01 * Xi_table[key, image, Xi_node + 1] +
                              h11 * Xi_tangent[key, image, Xi_node + 1]) * harmonics[key]
            end
            total_Xi += cis(dot(G, data.tau[image])) * amplitude
        end

        # Contract Δ with the shared Ξ sum, phased onto each molecule's position in the cell. Both
        # excitations of a block sit on the same molecule, so there is one phase for the whole block.
        weight = data.prefactor * exp(-(G_norm / (2 * parameters.eta))^2) / (G_norm * G_norm)
        for image in 1:n_images
            phase = cis(dot(G, data.tau[image]))
            for pair in 1:n_pairs
                row = Delta_row(image, pair, n_pairs)
                amplitude = zero(Complex{T})
                @simd for key in 1:size(Delta_table, 1)
                    amplitude += (h00 * Delta_table[key, row, Delta_node] +
                                  h10 * Delta_tangent[key, row, Delta_node] +
                                  h01 * Delta_table[key, row, Delta_node + 1] +
                                  h11 * Delta_tangent[key, row, Delta_node + 1]) * harmonics[key]
                end
                (s, t) = pairs.transitions_of_pair[pair]
                value = weight * phase * amplitude * conj(total_Xi)
                D_blocks[image, s, t] += value
                s == t && continue
                D_blocks[image, t, s] += conj(value)
            end
        end
    end

    # The G = 0 term, replaced by the dipole product scaled by 1/(2ε_b + 1). This is the boundary choice.
    # The 3/4π comes from writing the dipoles in terms of the ℓ = 1 coefficients, as in add_dipole_term!.
    boundary_weight = data.prefactor * T(3) / (T(4 * π) * (2 * data.epsilon_b + one(T)))
    # At ε_b = Inf the weight is exactly zero, which is the conducting boundary condition.
    if !iszero(boundary_weight)
        Delta_slopes = dipole_slope_matrix(rotated_Delta_bc, q_bc_invA, l_max)
        Xi_slopes = dipole_slope_matrix(rotated_Xi_bc, q_bc_invA, l_max)
        total_Xi_slopes = vec(sum(Xi_slopes, dims = 1))
        @inbounds for image in 1:n_images, pair in 1:n_pairs
            row = Delta_row(image, pair, n_pairs)
            total = zero(Complex{T})
            for component in 1:3
                total += Delta_slopes[row, component] * conj(total_Xi_slopes[component])
            end
            (s, t) = pairs.transitions_of_pair[pair]
            value = boundary_weight * total
            D_blocks[image, s, t] += value
            s == t && continue
            D_blocks[image, t, s] += conj(value)
        end
    end

    # The reciprocal lattice sum includes the same molecule at ΔR = 0, which D excludes, so subtract it.
    n_q = length(q_grid_invA)
    radial_weights = Vector{T}(undef, n_q)
    @inbounds for q_idx in 1:n_q
        # Endpoints use one directional difference, interior use central difference.
        step = q_idx == 1 ? q_grid_invA[2] - q_grid_invA[1] :
               q_idx == n_q ? q_grid_invA[n_q] - q_grid_invA[n_q - 1] :
               (q_grid_invA[q_idx + 1] - q_grid_invA[q_idx - 1]) / 2
        # Endpoint and interior weights are different from interior.
        radial_weights[q_idx] = (q_idx == 1 || q_idx == n_q ? step / 2 : step) *
                                exp(-(q_grid_invA[q_idx] / (2 * parameters.eta))^2)
    end
    # At d = 0 orthonormality collapses the angular integral to an inner product over keys.
    self_prefactor = Complex{T}(ALPHA_EM * HBAR_C_EV_ANGSTROM / (2 * π^2))
    @inbounds for image in 1:n_images, pair in 1:n_pairs
        row = Delta_row(image, pair, n_pairs)
        self = zero(Complex{T})
        for key in 1:n_keys, q_idx in 1:n_q
            self += radial_weights[q_idx] * rotated_Delta[row, q_idx, key] *
                    conj(rotated_Xi[image, q_idx, key])
        end
        (s, t) = pairs.transitions_of_pair[pair]
        value = self_prefactor * self
        D_blocks[image, s, t] -= value
        s == t && continue
        D_blocks[image, t, s] -= conj(value)
    end

    # Δ P̄ Ξ* is real for the real densities RKS gives us, so the imaginary part is round-off. Check
    # that before discarding it.
    scale = max(maximum(abs, real(D_blocks)), one(T))
    imaginary_residual = maximum(abs, imag(D_blocks))
    tolerance = T(1000) * eps(T) * scale
    imaginary_residual <= tolerance || error(
        "The Ewald long-range D has imaginary residual $(imaginary_residual), tolerance $(tolerance).")

    return real(D_blocks)
end

@inline function add_long_range!(
        H::Matrix{Complex{T}},
        data::EwaldLongRangeData{T},
        buffers::EwaldLongRangeBuffers{T},
        k::SVector{3, T},
    ) where {T<:AbstractFloat}
    """
    Add the reciprocal-space long-range coupling J_bar^LR_{λ_iλ_j′}(k) to the Bloch Hamiltonian.

    # Arguments:
    - H::Matrix{Complex{T}}: The Bloch Hamiltonian, whose upper triangle is accumulated into.
    - data::EwaldLongRangeData{T}: The k-independent long-range data.
    - buffers::EwaldLongRangeBuffers{T}: Per-task buffers.
    - k::SVector{3, T}: The Brillouin zone wavevector, in Å^{-1}.

    # Returns:
    - Nothing. Only the upper triangle of H is written.
    """

    # Get the dimensions.
    n_lambda_i = length(data.image_of)
    l_max_lr = data.l_max_lr
    n_keys_lr = size(data.f_table, 1)
    f_image = buffers.f_image
    harmonics = buffers.harmonics
    f_table = data.f_table
    f_tangent = data.f_tangent

    # Fold to the 1BZ.
    k_folded, _ = fold_to_bz(data.lattice, k)

    @inbounds for G in data.G_vectors
        # Find Q = k + G. The cutoff goes on |Q|, not |G|, because the Gaussian weight that sets
        # the truncation error is exp(-Q²/4η²).
        Q = k_folded + G
        Q_norm = norm(Q)
        Q_norm <= data.G_max || continue

        # Replace the Q = 0 term by the dipole product set by ε_b.
        if Q_norm < T(Q_ZERO_TOLERANCE)
            add_dipole_term!(H, data)
            continue
        end

        # Compute the spherical harmonics for this Q.
        theta = acos(clamp(Q[3] / Q_norm, -one(T), one(T)))
        phi = atan(Q[2], Q[1])
        computePlmcostheta!(buffers.Ylm_cache, theta, l_max_lr)
        computeYlm!(buffers.Ylm_cache, theta, phi, l_max_lr)
        Yvals = SphericalHarmonics.getY(buffers.Ylm_cache)

        # Flatten the harmonics.
        @inbounds for l in 0:l_max_lr
            key_base = l * l + l + 1
            for m in -l:l
                harmonics[key_base + m] = Complex{T}(Yvals[(l, m)])
            end
        end

        # Cubic Hermite stencil for |Q| on the uniform q grid.
        position = Q_norm / data.q_step
        node = min(Int(floor(position)) + 1, size(f_table, 3) - 1)
        s = position - (node - 1)
        s_squared = s * s
        s_cubed = s_squared * s
        h00 = 2 * s_cubed - 3 * s_squared + one(T)
        h10 = s_cubed - 2 * s_squared + s
        h01 = -2 * s_cubed + 3 * s_squared
        h11 = s_cubed - s_squared


        # Now compute f_{λ_i}(Q) = exp(i Q⋅τ) Σ_key f_key(Q) Y_key, with f interpolated by the Hermite stencil.
        @inbounds for lambda_i in 1:n_lambda_i
            total = zero(Complex{T})
            @simd for key in 1:n_keys_lr
                total += (h00 * f_table[key, lambda_i, node] +
                          h10 * f_tangent[key, lambda_i, node] +
                          h01 * f_table[key, lambda_i, node + 1] +
                          h11 * f_tangent[key, lambda_i, node + 1]) * harmonics[key]
            end
            f_image[lambda_i] = total
        end

        # Multiply by the image phases.
        @inbounds for image in eachindex(data.tau)
            buffers.image_phases[image] = cis(dot(Q, data.tau[image]))
        end
        @inbounds for lambda_i in 1:n_lambda_i
            f_image[lambda_i] *= buffers.image_phases[data.image_of[lambda_i]]
        end

        # Accumulate weight * f_image f_image^† into the upper triangle ('U') of H.
        weight = data.prefactor * exp(-(Q_norm / (2 * data.eta))^2) / (Q_norm * Q_norm)
        BLAS.her!('U', weight, f_image, H)
    end

    return nothing
end

@inline function add_dipole_term!(
        H::Matrix{Complex{T}},
        data::EwaldLongRangeData{T},
    ) where {T<:AbstractFloat}
    """
    Add the Q = 0 term of the long-range sum, which a sample surrounded by a medium of permittivity
    ε_b reduces to

        (μ_λ_i . conj(μ_{λ_j′})) / (2ε_b + 1).

    Each dipole is written in terms of the ℓ = 1 coefficients as μ_{λ_i,-m} = -i √(3/4π) (-1)^m lim f^{λ_i}_{1m}(q)/q.

    # Arguments:
    - H::Matrix{Complex{T}}: The Bloch Hamiltonian, whose upper triangle is accumulated into.
    - data::EwaldLongRangeData{T}: The k-independent long-range data.

    # Returns:
    - Nothing. Only the upper triangle of H is written.
    """

    # Get the dimensions.
    n_lambda_i = length(data.image_of)
    slopes = data.dipole_slopes
    weight = data.prefactor * T(3) / (T(4 * π) * (2 * data.epsilon_b + one(T)))

    # Nothing to add in the conducting limit, so skip the loop rather than adding zeros.
    iszero(weight) && return nothing

    @inbounds for lambda_j in 1:n_lambda_i, lambda_i in 1:lambda_j
        total = zero(Complex{T})
        # Sum (lim f^{λ_i}_{1m}(q)/q) conj(lim f^{λ_j′}_{1m}(q)/q) over the three m.
        for column in 1:3
            total += slopes[lambda_i, column] * conj(slopes[lambda_j, column])
        end
        H[lambda_i, lambda_j] += weight * total
    end

    return nothing
end

end
