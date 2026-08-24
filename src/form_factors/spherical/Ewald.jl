# This module implements the Ewald split of the intermolecular Coulomb coupling.

module Ewald

using StaticArrays
using SphericalHarmonics
using LinearAlgebra: norm, cross, dot, mul!, BLAS

using ..CrystalLattice: CrystalLatticeData, fold_to_bz, ALPHA_EM, HBAR_C_EV_ANGSTROM

export EwaldParameters, choose_ewald_parameters, short_range_kernel, supercell_radius,
       EwaldLongRangeData, EwaldLongRangeBuffers, build_ewald_long_range, add_long_range!, self_term_matrix

# Coefficients below this fraction of the largest one over the long-range q range are treated as
# absent, which sets the max ℓ in the long-range sum. With this set to 1e-4, the error is around
# 1e-10 eV, more or less zero. To be revisitied, and extended to the normal SCarFFF. 
const COEFFICIENT_THRESHOLD = 1.0e-4

# |Q| below this is taken as the Q = 0 term, which happens only at k = 0 with G = 0. Allows for
# rounding errors.
const Q_ZERO_TOLERANCE = 1.0e-10


struct EwaldParameters{T<:AbstractFloat}
    """
    The Ewald splitting parameter and the two cutoffs it implies.

    The cutoffs are pinned to a common truncation error ε on both ends, so that neither dominates:

        ε = exp(-η² R_max²) = exp(-Q_max²/4η²).

    Q_max and R_max are then chosen such that the amount of work done is minimised. To be revisited with
    cost measuring of both sums, and to make sure a requested ε is what we actually get.

    # Fields:
    - eta::T: The splitting parameter η, in Å^{-1}.
    - R_max::T: The real-space cutoff for the short-range sum, in Å. Widened to the 3x3x3 supercell
      radius when ε alone would ask for less, which only improves the truncation.
    - Q_max::T: The reciprocal-space cutoff for the long-range sum, in Å^{-1}.
    - epsilon::T: The truncation error the cutoffs are pinned to.
    """
    eta::T
    R_max::T
    Q_max::T
    epsilon::T
end

function choose_ewald_parameters(
        lattice::CrystalLatticeData{T},
        epsilon::Real;
        eta::Real = 0,
        cost_ratio::Real = 1,
    )::EwaldParameters{T} where {T<:AbstractFloat}
    """
    Choose η and the two cutoffs for a requested truncation error, ε.

    The two cutoffs are related by Q_max R_max = -2 ln ε, and the total work

        W = c_R N_R + c_Q N_Q,     N_R ≃ 4π R_max³ / 3 V_uc,   N_Q ≃ V_uc Q_max³ / 6π²,

    is minimised at

        R_max ≃ √(-2 ln ε) (V_uc² c_Q / 8π³ c_R)^{1/6},

    and η then satisfies η² = Q_max / 2R_max. We currently assume a cost ratio of 1.

    # Arguments:
    - lattice::CrystalLatticeData{T}: The crystal lattice, for the cell volume and reciprocal spacings.
    - epsilon::Real: The requested truncation error, in (0, 1).
    - eta::Real: A fixed η in Å^{-1}, or 0 to derive one.
    - cost_ratio::Real: c_Q / c_R, the cost of one reciprocal lattice vector relative to one real one.

    # Returns:
    - EwaldParameters{T}: The splitting parameter and both cutoffs.
    """

    # Check that both an error ∈ [0,1] and a cost ratio > 0 are given.
    zero(epsilon) < epsilon < one(epsilon) ||
        error("The Ewald truncation error must lie in (0, 1), got $(epsilon).")
    cost_ratio > 0 || error("The Ewald cost ratio must be positive, got $(cost_ratio).")

    # Extra factor of √2 since total error is √(ε_Q^2 + ε_R^2) = √2 ε.
    log_epsilon = -log(T(epsilon) / sqrt(T(2)))

    if eta > 0
        eta_value = T(eta)
    else
        # The work-minimising real-space cutoff.
        optimal_R = sqrt(2 * log_epsilon) * (lattice.volume^2 * T(cost_ratio) / (8 * T(π)^3))^(one(T) / 6)
        eta_value = sqrt(log_epsilon) / optimal_R
    end

    # Get the reciprocal-space cutoff using η.
    R_max = sqrt(log_epsilon) / eta_value
    Q_max = 2 * eta_value * sqrt(log_epsilon)

    # Now ensure that the real space cutoff includes at least the 3x3x3 supercell.
    R_max = max(R_max, supercell_radius(lattice))

    return EwaldParameters{T}(eta_value, R_max, Q_max, T(epsilon))
end

function supercell_radius(lattice::CrystalLatticeData{T})::T where {T<:AbstractFloat}
    """
    The distance to the furthest corner of the 3x3x3 supercell, which is the largest |ΔR| among the
    lattice vectors with all three indices in {-1, 0, 1}. This does not necessarily include all molecules in the 
    3x3x3 supercell in the short-range sum, and may include some outside, but instead pins the cutoff
    to a reasonable minimum.

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

    The form factor coefficients are held as a compact sub-table covering only q <= Q_max and
    ℓ <= l_max_lr, alongside the Hermite tangents used to interpolate them. That table is small
    enough to stay in cache, which matters because it is read at every q point of the outer grid.

    # Fields:
    - lattice::CrystalLatticeData{T}: The crystal lattice, used to fold k before summing.
    - G_vectors::Vector{SVector{3, T}}: Candidate reciprocal lattice vectors, in Å^{-1}.
    - f_table::Array{Complex{T}, 3}: The coefficients f̄, with dimensions (n_lambda, n_q_lr, n_keys_lr).
    - f_tangent::Array{Complex{T}, 3}: The Hermite tangents Δq df̄/dq, with the same dimensions.
    - q_step::T: The spacing of the (uniform) q grid, in Å^{-1}.
    - l_max_lr::Int: The ℓ ceiling of the sub-table.
    - translations::Vector{SVector{3, T}}: The distinct τ_{A,i}, one per molecule in the cell.
    - image_of::Vector{Int}: Which of those translations each λ carries, so its length is n_lambda.
    - dipole_slopes::Matrix{Complex{T}}: lim_{q→0} f̄_{1m}(q)/q for each λ, columns m = -1, 0, 1, in Å. Used for the boundary term.
    - self_term::Matrix{Complex{T}}: J^LR_{λλ'}(0) for λ, λ' on the same image, in eV.
    - eta::T: The splitting parameter η, in Å^{-1}.
    - Q_max::T: The reciprocal-space cutoff, in Å^{-1}.
    - prefactor::T: 4π α_EM ħc / V_uc, in eV Å^{-2}.
    - include_dipole_term::Bool: Boundary choice: whether the Q = 0 term is the dipole average or zero.
    """
    lattice::CrystalLatticeData{T}
    G_vectors::Vector{SVector{3, T}}
    f_table::Array{Complex{T}, 3}
    f_tangent::Array{Complex{T}, 3}
    q_step::T
    l_max_lr::Int
    translations::Vector{SVector{3, T}}
    image_of::Vector{Int}
    dipole_slopes::Matrix{Complex{T}}
    self_term::Matrix{Complex{T}}
    eta::T
    Q_max::T
    prefactor::T
    include_dipole_term::Bool
end

struct EwaldLongRangeBuffers{T<:AbstractFloat, C}
    """
    Per-task buffers for the long-range sum. One of these per BlochEigensystem, so it
    is allocated once per thread rather than once per q point.

    # Fields:
    - u::Vector{Complex{T}}: The phase-shifted molecular amplitudes u_λ(Q).
    - harmonics::Vector{Complex{T}}: Y_ℓ^μ(Q̂) flattened onto the table's key ordering.
    - image_phases::Vector{Complex{T}}: exp(i Q . τ) for each molecule in the cell.
    - Ylm_cache::C: The spherical harmonic cache, sized to the ℓ ceiling of the sub-table.
    """
    u::Vector{Complex{T}}
    harmonics::Vector{Complex{T}}
    image_phases::Vector{Complex{T}}
    Ylm_cache::C
end

function EwaldLongRangeBuffers(data::EwaldLongRangeData{T}) where {T<:AbstractFloat}
    cache = SphericalHarmonics.cache(data.l_max_lr, SphericalHarmonics.FullRange)
    return EwaldLongRangeBuffers{T, typeof(cache)}(
        Vector{Complex{T}}(undef, length(data.image_of)),
        Vector{Complex{T}}(undef, (data.l_max_lr + 1)^2),
        Vector{Complex{T}}(undef, length(data.translations)),
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
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients f̄, with dimensions (n_lambda, n_q, n_keys).
    - n_q_lr::Int: How many q points the long-range sum reaches.
    - l_max::Int: The maximum angular momentum mode of the form factor.

    # Returns:
    - Int: The ℓ ceiling.
    """

    n_lambda = size(rotated_R, 1)

    # Find the largest coefficient in each l block over the long-range q range.
    block_magnitude = zeros(T, l_max + 1)
    @inbounds for l in 0:l_max, key in (l * l + 1):((l + 1)^2), q_idx in 1:n_q_lr, lambda in 1:n_lambda
        block_magnitude[l + 1] = max(block_magnitude[l + 1], abs(rotated_R[lambda, q_idx, key]))
    end

    cutoff = T(COEFFICIENT_THRESHOLD) * maximum(block_magnitude)
    # Work backwards to find the first (largest ℓ) block with a magnitude greater than the cutoff. Keep everything below.
    for l in l_max:-1:0
        block_magnitude[l + 1] > cutoff && return l
    end

    return 0
end

function build_ewald_long_range(
        rotated_R::Array{Complex{T}, 3},
        tau_of_lambda::Vector{SVector{3, T}},
        image_of::Vector{Int},
        lattice::CrystalLatticeData{T},
        q_grid_invA::Vector{T},
        l_max::Int,
        parameters::EwaldParameters{T};
        include_dipole_term::Bool = true,
    )::EwaldLongRangeData{T} where {T<:AbstractFloat}
    """
    Assemble everything the long-range sum

        𝒥^LR_{λλ'}(k) = (4π α_EM ħc / V_uc) Σ_{Q ≠ 0}^{Q_max} (1/Q²) exp(-Q²/4η²) u_λ(Q) conj(u_{λ'}(Q)),
        
    with
        
        u_λ(Q) = exp(i Q . τ_{A,i}) Σ_{ℓμ} f̄^λ_{ℓμ}(Q) Y_ℓ^μ(Q̂),
        
    and Q = k + G,

    needs that does not depend on k: the G vectors, the interpolation table for f̄, the
    transition dipole slopes for the Q = 0 term, and the self-interaction that has to be subtracted.

    # Arguments:
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients f̄, with dimensions (n_lambda, n_q, n_keys).
    - tau_of_lambda::Vector{SVector{3, T}}: The translation τ_{A,i} for each λ, in Å. Only the
      distinct values are stored, since the phase depends on the molecule rather than on λ.
    - image_of::Vector{Int}: The image index for each λ, which fixes which pairs share a molecule.
    - lattice::CrystalLatticeData{T}: The crystal lattice.
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å. Must be uniform and start at zero.
    - l_max::Int: The maximum angular momentum mode of the form factor.
    - parameters::EwaldParameters{T}: The splitting parameter and cutoffs.
    - include_dipole_term::Bool: Whether to use the dipole average at Q = 0, or set it to zero, which
      is the conducting boundary condition.

    # Returns:
    - EwaldLongRangeData{T}: The k-independent long-range data.
    """

    # Get the dimensions.
    n_lambda = size(rotated_R, 1)
    n_q = length(q_grid_invA)

    # Check that we have enough points for interpolation.
    n_q >= 3 || error("The Ewald long-range sum needs at least three q points, got $(n_q).")
    q_step = q_grid_invA[2] - q_grid_invA[1]
    # The Hermite interpolation and the dipole slopes both assume a uniform grid anchored at zero.
    abs(q_grid_invA[1]) < eps(T) ||
        error("The Ewald long-range sum expects a q grid starting at zero, got $(q_grid_invA[1]).")
    # Use sqrt(eps) here, to make sure it is compatible with float32. This will still be well below
    # any meaningul deviations.
    uniform_tolerance = sqrt(eps(T)) * q_step
    all(abs(q_grid_invA[i + 1] - q_grid_invA[i] - q_step) < uniform_tolerance for i in 1:(n_q - 1)) ||
        error("The Ewald long-range sum expects a uniform q grid.")

    # Check that we can actually reach Q_max with the grid and error settings.
    q_grid_invA[end] >= parameters.Q_max || error(
        "The Ewald reciprocal cutoff Q_max = $(parameters.Q_max) Å^{-1} lies beyond the q grid, " *
        "which reaches $(q_grid_invA[end]) Å^{-1}. Raise q_max or allow for a larger truncation error."
    )

    # Trim to the q grid needed up to Q_max, ensuring we have the extra points for the interpolation.
    n_q_lr = min(n_q, Int(ceil(parameters.Q_max / q_step)) + 3)
    # Truncate l_max as well.
    l_max_lr = long_range_l_ceiling(rotated_R, n_q_lr, l_max)
    n_keys_lr = (l_max_lr + 1)^2

    # Store the subset of rotated coefficients that we need for the LR sum.
    f_table = Array{Complex{T}, 3}(undef, n_keys_lr, n_lambda, n_q_lr)
    @inbounds for q_idx in 1:n_q_lr, lambda in 1:n_lambda, key in 1:n_keys_lr
        f_table[key, lambda, q_idx] = rotated_R[lambda, q_idx, key]
    end

    # Also store the tangents for the Hermite interpolation. Premultiply by dx (i.e. don't divide by it) to save multiplying later.
    # Endpoints use one directional difference, interior points use central difference, hence the 1/2 for them.
    f_tangent = Array{Complex{T}, 3}(undef, n_keys_lr, n_lambda, n_q_lr)
    @inbounds for lambda in 1:n_lambda, key in 1:n_keys_lr
        f_tangent[key, lambda, 1] = f_table[key, lambda, 2] - f_table[key, lambda, 1]
        for q_idx in 2:(n_q_lr - 1)
            f_tangent[key, lambda, q_idx] =
                (f_table[key, lambda, q_idx + 1] - f_table[key, lambda, q_idx - 1]) / 2
        end
        f_tangent[key, lambda, n_q_lr] = f_table[key, lambda, n_q_lr] - f_table[key, lambda, n_q_lr - 1]
    end

    # Compute the G vectors.
    G_vectors = enumerate_reciprocal_vectors(lattice, parameters.Q_max + brillouin_zone_radius(lattice))

    # Compute the q -> 0 term, and self-interaction term.
    dipole_slopes = dipole_slope_matrix(rotated_R, q_grid_invA, l_max)
    self_term = self_term_matrix(rotated_R, image_of, q_grid_invA, l_max, parameters.eta)

    prefactor = T(4 * π * ALPHA_EM * HBAR_C_EV_ANGSTROM) / lattice.volume

    # Store the translations for each image.
    n_images = maximum(image_of)
    translations = [tau_of_lambda[findfirst(==(i), image_of)] for i in 1:n_images]

    return EwaldLongRangeData{T}(
        lattice, G_vectors, f_table, f_tangent, q_step, l_max_lr,
        translations, copy(image_of),
        dipole_slopes, self_term, parameters.eta, parameters.Q_max, prefactor, include_dipole_term,
    )
end

function dipole_slope_matrix(
        rotated_R::Array{Complex{T}, 3},
        q_grid_invA::Vector{T},
        l_max::Int,
    )::Matrix{Complex{T}} where {T<:AbstractFloat}
    """
    Extract lim_{q→0} f̄_{1m}(q)/q for every λ and m using a two point fit, f/q = a + b q.

    # Arguments:
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients f̄, with dimensions (n_lambda, n_q, n_keys).
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å.
    - l_max::Int: The maximum angular momentum mode of the form factor.

    # Returns:
    - Matrix{Complex{T}}: The slopes, with dimensions (n_lambda, 3) for m = -1, 0, 1, in Å.
    """

    l_max >= 1 || error("The Ewald Q = 0 term needs the ℓ = 1 coefficients, but l_max = $(l_max).")

    n_lambda = size(rotated_R, 1)
    slopes = Matrix{Complex{T}}(undef, n_lambda, 3)

    q_1 = q_grid_invA[2]
    q_2 = q_grid_invA[3]

    @inbounds for (column, m) in enumerate(-1:1)
        key = 1 * 1 + (1 + m) + 1
        for lambda in 1:n_lambda
            ratio_1 = rotated_R[lambda, 2, key] / q_1
            ratio_2 = rotated_R[lambda, 3, key] / q_2
            # Linear extrapolation of the ratio back to q = 0.
            slopes[lambda, column] = ratio_1 + (ratio_1 - ratio_2) * q_1 / (q_2 - q_1)
        end
    end

    return slopes
end

function self_term_matrix(
        rotated_R::Array{Complex{T}, 3},
        image_of::Vector{Int},
        q_grid_invA::Vector{T},
        l_max::Int,
        eta::T,
    )::Matrix{Complex{T}} where {T<:AbstractFloat}
    """
    The self-interaction J^LR_{λλ'}(0) that the reciprocal sum includes but the physics does not,
    since a molecule does not interact with itself. It is subtracted from the ΔR = 0 couplings.

    At zero separation the phase drops out and the full integral reduces to a single radial integral:

        J^LR_{λλ'}(0) = (α_EM ħc / 2π²) ∫dq exp(-q²/4η²) Σ_{ℓμ} f̄^λ_{ℓμ}(q) conj(f̄^{λ'}_{ℓμ}(q)).

    Only pairs on the same molecule are affected, so the matrix is zero between different images.

    # Arguments:
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients f̄, with dimensions (n_lambda, n_q, n_keys).
    - image_of::Vector{Int}: The image index for each λ.
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å.
    - l_max::Int: The maximum angular momentum mode of the form factor.
    - eta::T: The splitting parameter η, in Å^{-1}.

    # Returns:
    - Matrix{Complex{T}}: The self-interaction in eV, with dimensions (n_lambda, n_lambda).
    """

    # Get the dimensions.
    n_lambda = size(rotated_R, 1)
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
    self_term = zeros(Complex{T}, n_lambda, n_lambda)

    @inbounds for lambda_prime in 1:n_lambda, lambda in 1:lambda_prime
        # Skip pairs on different molecules.
        image_of[lambda] == image_of[lambda_prime] || continue

        total = zero(Complex{T})
        # Perform the integral.
        for key in 1:n_keys, q_idx in 1:n_q
            total += weights[q_idx] * rotated_R[lambda, q_idx, key] * conj(rotated_R[lambda_prime, q_idx, key])
        end

        value = prefactor * total
        self_term[lambda, lambda_prime] = value
        self_term[lambda_prime, lambda] = conj(value)
    end

    return self_term
end

@inline function add_long_range!(
        H::Matrix{Complex{T}},
        data::EwaldLongRangeData{T},
        buffers::EwaldLongRangeBuffers{T},
        k::SVector{3, T},
    ) where {T<:AbstractFloat}
    """
    Add the reciprocal-space long-range coupling 𝒥^LR_{λλ'}(k) to the Bloch Hamiltonian.

    # Arguments:
    - H::Matrix{Complex{T}}: The Bloch Hamiltonian, whose upper triangle is accumulated into.
    - data::EwaldLongRangeData{T}: The k-independent long-range data.
    - buffers::EwaldLongRangeBuffers{T}: Per-task buffers.
    - k::SVector{3, T}: The Brillouin zone wavevector, in Å^{-1}.

    # Returns:
    - Nothing. Only the upper triangle of H is written.
    """

    # Get the dimensions.
    n_lambda = length(data.image_of)
    l_max_lr = data.l_max_lr
    n_keys_lr = size(data.f_table, 1)
    u = buffers.u
    harmonics = buffers.harmonics
    f_table = data.f_table
    f_tangent = data.f_tangent

    # Fold to the 1BZ.
    k_folded, _ = fold_to_bz(data.lattice, k)

    @inbounds for G in data.G_vectors
        # Find Q = k + G.
        Q = k_folded + G
        Q_norm = norm(Q)
        Q_norm <= data.Q_max || continue

        # Use the dipole BC if specified.
        if Q_norm < T(Q_ZERO_TOLERANCE)
            data.include_dipole_term && add_dipole_term!(H, data)
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


        # Now compute u_λ(Q) = exp(i Q⋅τ) Σ_key f̄_key(Q) Y_key, with f̄ interpolated by the Hermite stencil. 
        @inbounds for lambda in 1:n_lambda
            total = zero(Complex{T})
            @simd for key in 1:n_keys_lr
                total += (h00 * f_table[key, lambda, node] +
                          h10 * f_tangent[key, lambda, node] +
                          h01 * f_table[key, lambda, node + 1] +
                          h11 * f_tangent[key, lambda, node + 1]) * harmonics[key]
            end
            u[lambda] = total
        end

        # Multiply by the image phases.
        @inbounds for image in eachindex(data.translations)
            buffers.image_phases[image] = cis(dot(Q, data.translations[image]))
        end
        @inbounds for lambda in 1:n_lambda
            u[lambda] *= buffers.image_phases[data.image_of[lambda]]
        end

        # Accumulate weight * u u^† into the upper triangle ('U') of H.
        weight = data.prefactor * exp(-(Q_norm / (2 * data.eta))^2) / (Q_norm * Q_norm)
        BLAS.her!('U', weight, u, H)
    end

    return nothing
end

@inline function add_dipole_term!(
        H::Matrix{Complex{T}},
        data::EwaldLongRangeData{T},
    ) where {T<:AbstractFloat}
    """
    Add the Q = 0 term of the long-range sum, which the angular average ⟨Q̂_u Q̂_v⟩ = δ_uv/3 reduces to

        (1/3)(μ_λ . conj(μ_{λ'})) = (1/4π) Σ_m [lim f̄^λ_{1m}(q)/q] conj[lim f̄^{λ'}_{1m}(q)/q].

    This is just one choice of boundary condition. The conducting one sets it to zero instead.

    # Arguments:
    - H::Matrix{Complex{T}}: The Bloch Hamiltonian, whose upper triangle is accumulated into.
    - data::EwaldLongRangeData{T}: The k-independent long-range data.

    # Returns:
    - Nothing. Only the upper triangle of H is written.
    """

    # Get the dimensions.
    n_lambda = length(data.image_of)
    slopes = data.dipole_slopes
    weight = data.prefactor / T(4 * π)

    @inbounds for lambda_prime in 1:n_lambda, lambda in 1:lambda_prime
        total = zero(Complex{T})
        # Sum 1/4π * (lim f̄^λ_{1m}(q)/q) conj(lim f̄^{λ'}_{1m}(q)/q).
        for column in 1:3
            total += slopes[lambda, column] * conj(slopes[lambda_prime, column])
        end
        H[lambda, lambda_prime] += weight * total
    end

    return nothing
end

end
