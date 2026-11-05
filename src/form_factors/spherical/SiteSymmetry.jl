# This module finds the site symmetry matrices ζ.

module SiteSymmetry

using StaticArrays
using LinearAlgebra: inv, dot
using SphericalHarmonics

using ..CrystalSymmetry: SymmetryOperation, cartesian_operation
using ..BlochHamiltonian: CrystalImage, CrystalExcitationBasis

export SiteSymmetryTransfer, build_site_symmetry, site_symmetry_operation

# Transitions whose energies sit closer than this are treated as one degenerate set, so that ζ is
# allowed to mix them.
const DEGENERACY_TOLERANCE = 0.01 # eV

# How many directions ζ is fitted on. This just needs to be big enough for the largest degenerate manifold,
# since the relation should hold at all q.
const N_PROBE_DIRECTIONS = 12

# A matrix element smaller than this counts as zero, so an unmixed state keeps an exact sign.
const ZETA_THRESHOLD = 1.0e-3

struct SiteSymmetryTransfer{T<:AbstractFloat}
    """
    One term of C_{(A_{j'},s')}(k_Λ) = e^{-i k_Λ · L} Σ_s ζ^{(s,s')} C_{(A_j,s)}(k).

    Written out as a list of terms rather than as a matrix because ζ is nearly always diagonal, so
    a sparse form is both the common case and the cheapest to apply.

    # Fields:
    - source::Int: The λ_i the coefficient is taken from.
    - target::Int: The λ_i it contributes to.
    - weight::T: The matrix element ζ^{(s,s')}.
    """
    source::Int
    target::Int
    weight::T
end

function site_symmetry_operation(
        operation::SymmetryOperation{T},
        images::Vector{CrystalImage{T}},
        image_index::Int,
        orientations::Vector{SMatrix{3, 3, T, 9}},
    )::Tuple{SMatrix{3, 3, T, 9}, SVector{3, T}} where {T<:AbstractFloat}
    """
    Find the operation h that an image picks up when a crystal symmetry puts it back in place.

    # Arguments:
    - operation::SymmetryOperation{T}: The crystal symmetry operation Λ_i.
    - images::Vector{CrystalImage{T}}: The molecules in the unit cell.
    - image_index::Int: Which image j is being carried.
    - orientations::Vector{SMatrix{3, 3, T, 9}}: Each image's rotation R_j, including its sign.

    # Returns:
    - Tuple: The rotation and the translation of h in image j's frame, the latter in Å.
    """

    j = image_index
    j_prime = operation.image_of[j]
    to_image = orientations[j] * inv(orientations[j_prime])

    rotation = to_image * operation.R_Lambda
    translation = to_image * (operation.R_Lambda * images[j].tau + operation.tau_Lambda -
                              images[j_prime].tau - operation.L_offsets[j])
    return rotation, translation
end

function degenerate_sets(energies::Vector{T})::Vector{Vector{Int}} where {T<:AbstractFloat}
    """
    Group the transitions whose energies are equal to within the degeneracy tolerance.

    # Arguments:
    - energies::Vector{T}: The monomer excitation energies, in eV.

    # Returns:
    - Vector{Vector{Int}}: The transitions in each set, which together cover every transition.
    """

    order = sortperm(energies)
    sets = Vector{Int}[]
    for index in order
        if !isempty(sets) && abs(energies[index] - energies[last(sets[end])]) < T(DEGENERACY_TOLERANCE)
            push!(sets[end], index)
        else
            push!(sets, [index])
        end
    end
    return sets
end

function probe_directions(::Type{T}, count::Int)::Vector{SVector{3, T}} where {T<:AbstractFloat}
    """
    Spread a few directions over the sphere, for evaluating the form factor on.

    They come from the golden spiral, so they are well separated whatever the count, and they are
    the same every run.

    # Arguments:
    - T::Type: The floating point type.
    - count::Int: How many directions.

    # Returns:
    - Vector{SVector{3, T}}: Unit vectors.
    """

    # Compute the golden angle, 2π/φ^2.
    golden = T(π) * (3 - sqrt(T(5)))
    directions = Vector{SVector{3, T}}(undef, count)

    # Build the directions.
    for index in 1:count
        z = one(T) - 2 * (index - T(0.5)) / count
        radius = sqrt(max(zero(T), one(T) - z^2))
        angle = golden * (index - 1)
        directions[index] = SVector{3, T}(radius * cos(angle), radius * sin(angle), z)
    end
    return directions
end

function form_factor_at(
        R_tensor::Array{Complex{T}, 3},
        transition::Int,
        q_index::Int,
        direction::SVector{3, T},
        l_max::Int,
        cache,
    )::Complex{T} where {T<:AbstractFloat}
    """
    Evaluate one transition's monomer form factor in a given direction.

    # Arguments:
    - R_tensor::Array{Complex{T}, 3}: The conformer's coefficients, (transition, q, key).
    - transition::Int: Which transition.
    - q_index::Int: Which point of the radial grid.
    - direction::SVector{3, T}: The direction q̂, a unit vector.
    - l_max::Int: The largest angular mode.
    - cache: A spherical harmonic cache to evaluate into.

    # Returns:
    - Complex{T}: f_s(q).
    """

    # Compute the spherical harmonics for one direction.
    theta = acos(clamp(direction[3], -one(T), one(T)))
    phi = atan(direction[2], direction[1])
    computePlmcostheta!(cache, theta, l_max)
    computeYlm!(cache, theta, phi, l_max)
    harmonics = SphericalHarmonics.getY(cache)

    # Turn the R tensor into the form factor.
    total = zero(Complex{T})
    @inbounds for key in 1:((l_max + 1)^2)
        total += R_tensor[transition, q_index, key] * harmonics[key]
    end
    return total
end

function site_symmetry_matrix(
        rotated_R::Array{Complex{T}, 3},
        lambda_i_of_transition::Vector{Int},
        energies::Vector{T},
        q_grid_invA::Vector{T},
        l_max::Int,
        R_h::SMatrix{3, 3, T, 9},
        tau_h::SVector{3, T},
    )::Matrix{T} where {T<:AbstractFloat}
    """
    Solve for ζ from the monomer form factor, one degenerate set at a time.

    # Arguments:
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients, (λ_i, q, key).
    - lambda_i_of_transition::Vector{Int}: The λ_i holding each transition of the image used.
    - energies::Vector{T}: The monomer excitation energies, in eV, which fix the degenerate sets.
    - q_grid_invA::Vector{T}: The radial grid, in Å^-1.
    - l_max::Int: The largest angular mode.
    - R_h, tau_h: The site symmetry operation, in that image's frame.

    # Returns:
    - Matrix{T}: ζ^{(s,s')}, indexed [s, s'].
    """

    # Get the dimensions, and build the ζ matrix.
    n_transitions = length(lambda_i_of_transition)
    zeta = zeros(T, n_transitions, n_transitions)

    # Skip q = 0, where every direction gives the same value and the relation says nothing.
    q_index = max(2, cld(length(q_grid_invA), 3))
    q_magnitude = q_grid_invA[q_index]
    directions = probe_directions(T, N_PROBE_DIRECTIONS)
    cache = SphericalHarmonics.cache(l_max, SphericalHarmonics.FullRange)
    inverse_R_h = inv(R_h)

    for set in degenerate_sets(energies)
        # The right hand side, f_{s'}(q) for every member of the set, at each probe direction.
        right = Matrix{Complex{T}}(undef, length(directions), length(set))
        for (row, direction) in enumerate(directions), (column, s_prime) in enumerate(set)
            right[row, column] = form_factor_at(rotated_R, lambda_i_of_transition[s_prime],
                                                q_index, direction, l_max, cache)
        end

        for s in set
            # The left hand side, e^{i q · τ_h} f_s(R_h^{-1} q).
            left = Vector{Complex{T}}(undef, length(directions))
            for (row, direction) in enumerate(directions)
                q_vector = q_magnitude * direction
                rotated = SVector{3, T}(inverse_R_h * direction)
                left[row] = cis(dot(q_vector, tau_h)) *
                            form_factor_at(rotated_R, lambda_i_of_transition[s],
                                           q_index, rotated, l_max, cache)
            end

            # ζ is real, so fit the real and imaginary parts together rather than solving a complex
            # system that would let it drift off the real axis.
            design = vcat(real.(right), imag.(right))
            observed = vcat(real.(left), imag.(left))
            solution = design \ observed

            for (column, s_prime) in enumerate(set)
                value = solution[column]
                zeta[s, s_prime] = abs(value) < T(ZETA_THRESHOLD) ? zero(T) : value
            end
        end
    end
    return zeta
end

function build_site_symmetry(
        basis::CrystalExcitationBasis{T},
        operations::Vector{SymmetryOperation{T}},
        rotated_R::Array{Complex{T}, 3},
        q_grid_invA::Vector{T},
        l_max::Int,
    )::Vector{Vector{SiteSymmetryTransfer{T}}} where {T<:AbstractFloat}
    """
    Work out how each symmetry operation moves the mixing coefficients between λ_i.

    # Arguments:
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - operations::Vector{SymmetryOperation{T}}: The crystal symmetry operations.
    - rotated_R::Array{Complex{T}, 3}: The rotated coefficients, (λ_i, q, key).
    - q_grid_invA::Vector{T}: The radial grid, in Å^-1.
    - l_max::Int: The largest angular mode.

    # Returns:
    - Vector{Vector{SiteSymmetryTransfer{T}}}: The terms to apply for each operation.
    """

    # Each image's own rotation R_j, which is what takes the reference molecule to that image.
    # Each image's rotation R_j.
    orientations = [cartesian_operation(image) for image in basis.images]
    transfers = Vector{Vector{SiteSymmetryTransfer{T}}}(undef, length(operations))

    # Track which of the unique site symmetry operations have already been solved, so that the same h is not solved again.
    solved = Dict{Tuple{Int, UInt64}, Matrix{T}}()

    for (operation_index, operation) in enumerate(operations)
        terms = SiteSymmetryTransfer{T}[]
        for image_index in eachindex(basis.images)
            # This image's λ_i, and the monomer energies that set the degenerate sets.
            lambda_i_of_transition = [basis.lambda_i_of[image_index, s]
                                      for s in 1:basis.n_transitions]
            energies = [basis.Delta_E_monomer[lambda_i] for lambda_i in lambda_i_of_transition]

            # The site operation h this image picks up under Λ.
            R_h, tau_h = site_symmetry_operation(operation, basis.images, image_index, orientations)

            # Round R_h and τ_h before hashing, so two h that agree give one key.
            key = (image_index, hash(round.(vcat(vec(R_h), tau_h), digits = 6)))
            zeta = get!(solved, key) do
                site_symmetry_matrix(rotated_R, lambda_i_of_transition, energies,
                                     q_grid_invA, l_max, R_h, tau_h)
            end

            # One term per non-zero ζ, carrying (image, s) onto (Λ(image), s').
            target_image = operation.image_of[image_index]
            for s in 1:basis.n_transitions, s_prime in 1:basis.n_transitions
                iszero(zeta[s, s_prime]) && continue
                push!(terms, SiteSymmetryTransfer{T}(
                    basis.lambda_i_of[image_index, s],
                    basis.lambda_i_of[target_image, s_prime],
                    zeta[s, s_prime]))
            end
        end
        transfers[operation_index] = terms
    end
    return transfers
end

end
