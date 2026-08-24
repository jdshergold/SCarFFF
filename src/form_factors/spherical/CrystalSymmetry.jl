# This module finds the symmetry operations that let one Bloch diagonalisation cover the whole
# star of k points.
#
# Two relations are used, both derived in crystal_upgrade.tex §"Some symmetries" and both verified
# against the assembled H(k) to round off:
#
#   Time reversal.  J is real, since it is a Coulomb integral over real transition densities and a
#   real separation, so H(k) = conj(H(-k)). This holds for any crystal and needs no crystallography.
#
#   The star of k.  For a space group element Λ with rotation R_Λ, carrying molecule A_i onto
#   molecule Λ(A_i) up to a lattice vector L_{Λ,A_i}, the eigenvectors at k_Λ = R_Λ^{-1} k follow
#   from those at k by a permutation and a phase,
#
#       E_Ψ(k_Λ) = E_Ψ(k),    C_{Λ(A_i),s}(k_Λ) = C_{A_i,s}(k) exp(-i k_Λ · L_{Λ,A_i}).
#
# Composing the two gives what the reconstruction applies, for a member whose own wavevector is k_m.
# Without time reversal,
#
#       C_{Λ(A_i),s}(k_m) =      C_{A_i,s}(k)  exp(-i k_m · L_{Λ,A_i}),
#
# and with it, the same thing on the conjugated coefficients,
#
#       C_{Λ(A_i),s}(k_m) = conj(C_{A_i,s}(k)) exp(-i k_m · L_{Λ,A_i}).
#
# Writing k_m rather than k_Λ is what lets the two share a phase factor: the two differ by a
# reciprocal lattice vector at most, and G · L is a multiple of 2π, so the phase cannot tell them
# apart. The operations come from the images already recorded in the crystal metadata, so no space
# group library and no metadata change is needed.

module CrystalSymmetry

using StaticArrays
using Quaternionic
using LinearAlgebra: inv

using ..CrystalLattice: CrystalLatticeData
using ..BlochHamiltonian: CrystalImage, CrystalExcitationBasis

export SymmetryOperation, StarMember, Stars, derive_symmetry_operations,
       build_stars, star_reduction_factor, choose_compatible_phi_count

# How close a matched rotation, or a lattice vector's fractional coordinates, has to be.
# Max part is used to make sure float32 is not too strict to work.
match_tolerance(::Type{T}) where {T<:AbstractFloat} = max(T(1.0e-6), 100 * eps(T))

# Decimal places used to recognise two grid directions as the same. Float32 needs some slack.
direction_digits(::Type{Float32}) = 5
direction_digits(::Type{T}) where {T<:AbstractFloat} = 8

struct SymmetryOperation{T<:AbstractFloat}
    """
    One operation of the star.

    # Fields:
    - rotation::SMatrix{3, 3, T, 9}: The Cartesian rotation R_Λ, including the improper sign.
    - image_of::Vector{Int}: Λ(A_i), the image each image is carried onto.
    - offsets::Vector{SVector{3, T}}: The lattice vector L_{Λ,A_i} left over for each image, in Å.
    - conjugate::Bool: Whether this operation is composed with time reversal, which sends k to -k
      and conjugates the coefficients.
    """
    rotation::SMatrix{3, 3, T, 9}
    image_of::Vector{Int}
    offsets::Vector{SVector{3, T}}
    conjugate::Bool
end

struct StarMember{T<:AbstractFloat}
    """
    One direction reached from an star's representative, and where its results belong.

    # Fields:
    - operation::Int: Index of an operation that maps k onto k_Λ.
    - direction::SVector{3, T}: The direction of the member. This is k_Λ.
    - grid_indices::Vector{Tuple{Int, Int}}: Every (θ, ϕ) index that points in this direction. The poles and
      the repeated ϕ endpoint mean one direction can sit at several indices, and each still needs
      its own entry in the output grid.
    """
    operation::Int
    direction::SVector{3, T}
    grid_indices::Vector{Tuple{Int, Int}}
end

struct Stars{T<:AbstractFloat}
    """
    The grid directions grouped into stars, so that each star needs only one diagonalisation.

    # Fields:
    - irreducible_directions::Vector{SVector{3, T}}: The unit vector of each star representative.
    - members::Vector{Vector{StarMember{T}}}: Per representative, every direction in its star,
      including the representative itself under the identity.
    - operations::Vector{SymmetryOperation{T}}: The operations that survived the grid closure test.
    """
    irreducible_directions::Vector{SVector{3, T}}
    members::Vector{Vector{StarMember{T}}}
    operations::Vector{SymmetryOperation{T}}
end

function choose_compatible_phi_count(
        n_theta::Int,
        n_phi::Int,
        operations::Vector{SymmetryOperation{T}};
        max_extra::Int = 12,
    )::Int where {T<:AbstractFloat}
    """
    The smallest ϕ grid, at least as fine as the one asked for, on which the most symmetry operations
    close. Why do we do this? The ϕ grid decides how many operations are usable, because ϕ + π (time reversal) and the
    rotation angles have to land on grid nodes. In general, this means we need N_ϕ-1 even, and N_ϕ-1 mod n = 0, with
    n each of the rotation orders in the symmetry group.

    For now this catches most symops, but misses some e.g. cubic and trigonal. To be revisited using a better symmetry
    adapted grid.

    # Arguments:
    - n_theta::Int: The number of θ grid points, which does not constrain anything but is needed to
      run the test.
    - n_phi::Int: The number of ϕ grid points asked for, and the lower bound on the answer.
    - operations::Vector{SymmetryOperation{T}}: The candidate operations.
    - max_extra::Int: How many extra ϕ points to consider (default: 12).

    # Returns:
    - Int: The chosen number of ϕ grid points.
    """

    theta_grid = collect(range(zero(T), T(π), length = n_theta))

    best_count = -1
    best_n_phi = n_phi
    # Keep trying different grid sizes until we find one that works. As a mod 12 grid fits everything
    # except for trigonal and hexagonal, this will take at most 12 tries.
    for candidate in n_phi:(n_phi + max_extra)
        phi_grid = collect(range(zero(T), T(2π), length = candidate))
        count = length(build_stars(theta_grid, phi_grid, operations).operations)
        if count > best_count
            best_count = count
            best_n_phi = candidate
            # Nothing can beat every operation closing, so stop as soon as that happens.
            count == length(operations) && break
        end
    end

    return best_n_phi
end

function star_reduction_factor(stars::Stars, n_theta::Int, n_phi::Int)::Float64
    """
    How many times fewer diagonalisations the stars buy, against one per grid point.
    Purely for reporting purposes.

    # Arguments:
    - stars::Stars: The stars.
    - n_theta::Int: The number of θ grid points.
    - n_phi::Int: The number of ϕ grid points.

    # Returns:
    - Float64: The ratio of grid points to representatives.
    """

    return n_theta * n_phi / length(stars.irreducible_directions)
end

function cartesian_operation(image::CrystalImage{T})::SMatrix{3, 3, T, 9} where {T<:AbstractFloat}
    """
    The full rotation operation for an image: the proper rotation times the improper sign.

    # Arguments:
    - image::CrystalImage{T}: The image.

    # Returns:
    - SMatrix{3, 3, T, 9}: The operation R_i, with det(R_i) = det_rotation.
    """

    return SMatrix{3, 3, T, 9}(image.det_rotation .* Quaternionic.to_rotation_matrix(image.rotation))
end

function is_lattice_vector(lattice::CrystalLatticeData{T}, vector::SVector{3, T})::Bool where {T<:AbstractFloat}
    """
    Check whether a displacement is a lattice vector, tested in fractional coordinates.

    # Arguments:
    - lattice::CrystalLatticeData{T}: The crystal lattice.
    - vector::SVector{3, T}: The displacement, in Å.

    # Returns:
    - Bool: True if the fractional coordinates are integers to within MATCH_TOLERANCE.
    """

    # The rows of `direct` are the lattice vectors, so the fractional coordinates satisfy Aᵀ x = vector.
    fractional = inv(transpose(lattice.direct)) * vector

    return all(component -> abs(component - round(component)) < match_tolerance(T), fractional)
end

function match_operation(
        rotation::SMatrix{3, 3, T, 9},
        translation::SVector{3, T},
        operations::Vector{SMatrix{3, 3, T, 9}},
        translations::Vector{SVector{3, T}},
        lattice::CrystalLatticeData{T},
    )::Union{Nothing, SymmetryOperation{T}} where {T<:AbstractFloat}
    """
    Work out where a candidate operation sends each image, or decide that it is not a symmetry.

    Applying (R, τ_Λ) to image i gives a molecule with orientation R M_i sitting at R τ_i + τ_Λ. For
    the operation to be a symmetry of the crystal that has to be one of the images we already have,
    up to a lattice vector, and both conditions have to hold for the same image. 

    # Arguments:
    - rotation::SMatrix{3, 3, T, 9}: The candidate Cartesian rotation R.
    - translation::SVector{3, T}: The candidate translation τ_Λ, in Å.
    - operations::Vector{SMatrix{3, 3, T, 9}}: The Cartesian operation of each image.
    - translations::Vector{SVector{3, T}}: The translation of each image, in Å.
    - lattice::CrystalLatticeData{T}: The crystal lattice.

    # Returns:
    - Union{Nothing, SymmetryOperation{T}}: The operation with its image map and offsets, or nothing
      if it is not a symmetry.
    """

    # Get the dimensions.
    n_images = length(operations)
    image_of = zeros(Int, n_images)
    offsets = Vector{SVector{3, T}}(undef, n_images)

    for i in 1:n_images
        # Here operations is the "orientation" of molecule A_i. Apply the crystal rotation to it, to see where it points after.
        rotated = rotation * operations[i]
        position = rotation * translations[i] + translation

        for j in 1:n_images
            # Check if each image points along the same direction as the rotated one.
            maximum(abs, rotated - operations[j]) < match_tolerance(T) || continue
            # Check that the rotated image sits on top of the candidate image, up to a lattice vector.
            offset = position - translations[j]
            is_lattice_vector(lattice, offset) || continue
            image_of[i] = j
            offsets[i] = offset
            break
        end

        image_of[i] == 0 && return nothing
    end

    return SymmetryOperation{T}(rotation, image_of, offsets, false)
end

function derive_symmetry_operations(
        images::Vector{CrystalImage{T}},
        lattice::CrystalLatticeData{T},
    )::Vector{SymmetryOperation{T}} where {T<:AbstractFloat}
    """
    Find the symmetry operations relating the images in the unit cell, each also paired with time
    reversal.

    # Arguments:
    - images::Vector{CrystalImage{T}}: The molecules in the unit cell.
    - lattice::CrystalLatticeData{T}: The crystal lattice.

    # Returns:
    - Vector{SymmetryOperation{T}}: The operations, identity first, then their time reversed partners.
    """

    # Setup the lists of rotations and translations.
    n_images = length(images)
    operations = [cartesian_operation(image) for image in images]
    translations = [image.translation for image in images]

    # The recorded pairs orient and place each image relative to the reference conformer.
    # As a result, to get from image e.g. 1 to i, we need to map 1 -> reference, then
    # reference -> i. This means we need R_i R_1^{-1} and τ_i - R_i R_1^{-1} τ_1, which is what the following does.
    reference_inverse = inv(operations[1])
    candidates = Tuple{SMatrix{3, 3, T, 9}, SVector{3, T}}[]
    for i in 1:n_images
        rotation = operations[i] * reference_inverse
        push!(candidates, (rotation, translations[i] - rotation * translations[1]))
    end

    # Close the space group, if it is not already. Usually does nothing.
    growing = true
    while growing
        growing = false
        for (rotation_a, translation_a) in copy(candidates),
            (rotation_b, translation_b) in copy(candidates)

            composed_rotation = rotation_a * rotation_b
            composed_translation = rotation_a * translation_b + translation_a
            seen = any(candidates) do candidate
                maximum(abs, candidate[1] - composed_rotation) < match_tolerance(T) &&
                    is_lattice_vector(lattice, candidate[2] - composed_translation)
            end
            if !seen
                push!(candidates, (composed_rotation, composed_translation))
                growing = true
            end
        end
    end

    found = SymmetryOperation{T}[]
    # Build the symops: find which image each operation maps each image onto, and the lattice vector left over.
    for (rotation, translation) in candidates
        operation = match_operation(rotation, translation, operations, translations, lattice)
        operation === nothing && continue
        push!(found, operation)
    end

    # Put the idenity operation first in the list.
    identity_position = findfirst(
        operation -> maximum(abs, operation.rotation - one(SMatrix{3, 3, T, 9})) < match_tolerance(T), found)
    identity_position === nothing && error(
        "The identity is not among the matched symmetry operations, which should be impossible. " *
        "Check the translations and rotations in the crystal metadata."
    )
    if identity_position != 1
        found[1], found[identity_position] = found[identity_position], found[1]
    end

    # Add in the time reversed operations, k -> -k and conjugation.
    time_reversed = [SymmetryOperation{T}(operation.rotation, operation.image_of, operation.offsets, true)
                     for operation in found]

    return vcat(found, time_reversed)
end

function build_stars(
        theta_grid::Vector{T},
        phi_grid::Vector{T},
        operations::Vector{SymmetryOperation{T}},
    )::Stars{T} where {T<:AbstractFloat}
    """
    Group the grid directions into stars under the operations that map the grid onto itself.

    # Arguments:
    - theta_grid::Vector{T}: The θ grid, in radians.
    - phi_grid::Vector{T}: The ϕ grid, in radians.
    - operations::Vector{SymmetryOperation{T}}: The candidate operations.

    # Returns:
    - Stars{T}: The stars, and the operations that survived.
    """

    # Get the grid dimensions.
    n_theta = length(theta_grid)
    n_phi = length(phi_grid)

    # Get the direction vector.
    function direction_at(theta_idx::Int, phi_idx::Int)
        sin_theta, cos_theta = sincos(theta_grid[theta_idx])
        sin_phi, cos_phi = sincos(phi_grid[phi_idx])
        return SVector{3, T}(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)
    end

    # Define a key for each direction, to group them into stars.
    digits = direction_digits(T)
    direction_key(v) = ntuple(i -> round(v[i] + zero(T), digits = digits) + zero(T), 3)

    # Collapse the grid onto distinct directions, remembering every index each one occupies.
    key_to_direction = Dict{NTuple{3, T}, Int}()
    direction_vectors = SVector{3, T}[]
    direction_indices = Vector{Tuple{Int, Int}}[]
    direction_at_node = Matrix{Int}(undef, n_theta, n_phi)
    for theta_idx in 1:n_theta, phi_idx in 1:n_phi
        vector = direction_at(theta_idx, phi_idx)
        key = direction_key(vector)
        index = get(key_to_direction, key, 0)
        if index == 0
            push!(direction_vectors, vector)
            push!(direction_indices, Tuple{Int, Int}[])
            index = length(direction_vectors)
            key_to_direction[key] = index
        end
        push!(direction_indices[index], (theta_idx, phi_idx))
        direction_at_node[theta_idx, phi_idx] = index
    end
    n_directions = length(direction_vectors)

    # Look a rotated direction up by snapping it to the nearest grid node and checking the residual.
    residual_tolerance = max(sqrt(eps(T)), T(1.0e-8))
    function nearest_direction(w::SVector{3, T})
        # Get the angles.
        theta = acos(clamp(w[3], -one(T), one(T)))
        phi = mod(atan(w[2], w[1]), T(2π))
        theta_idx = clamp(searchsortedfirst(theta_grid, theta), 1, n_theta)
        if theta_idx > 1 && abs(theta_grid[theta_idx - 1] - theta) < abs(theta_grid[theta_idx] - theta)
            theta_idx -= 1
        end
        phi_idx = clamp(searchsortedfirst(phi_grid, phi), 1, n_phi)
        if phi_idx > 1 && abs(phi_grid[phi_idx - 1] - phi) < abs(phi_grid[phi_idx] - phi)
            phi_idx -= 1
        end
        # The residual is what decides it, so a near miss on the index costs nothing.
        maximum(abs, direction_at(theta_idx, phi_idx) - w) <= residual_tolerance || return 0
        return direction_at_node[theta_idx, phi_idx]
    end

    # An operation is usable only if every direction has an image among the directions.
    usable = SymmetryOperation{T}[]
    image_tables = Vector{Vector{Int}}()
    for operation in operations
        inverse = inv(operation.rotation)
        sign = operation.conjugate ? -one(T) : one(T) # Change sign if this is a time reversed operation.
        table = Vector{Int}(undef, n_directions)
        closed = true
        for index in 1:n_directions
            # Find the image of the direction: sign * R^{-1} * direction.
            target = nearest_direction(sign .* (inverse * direction_vectors[index]))
            if target == 0
                closed = false
                break
            end
            table[index] = target
        end
        if closed
            push!(usable, operation)
            push!(image_tables, table)
        end
    end

    # The surviving operations are closed under composition and inversion, so applying all of them
    # to a representative enumerates its whole star directly.
    owner = zeros(Int, n_directions)
    irreducible_directions = SVector{3, T}[]
    members = Vector{StarMember{T}}[]

    for index in 1:n_directions
        # If a direction is unclaimed (owner == 0), it becomes the representative of its star.
        owner[index] == 0 || continue

        push!(irreducible_directions, direction_vectors[index])
        star_index = length(irreducible_directions)
        star = StarMember{T}[]

        # Now go through the table for this representative, and claim every direction.
        for (operation_index, table) in enumerate(image_tables)
            target = table[index]
            owner[target] == 0 || continue
            owner[target] = star_index
            push!(star, StarMember{T}(operation_index, direction_vectors[target],
                                        direction_indices[target]))
        end

        push!(members, star)
    end

    # Every direction must end up in exactly one star, so check it rather than trust the group structure.
    all(!iszero, owner) || error(
        "$(count(iszero, owner)) of $(n_directions) grid directions were left out of every star, " *
        "which means the symmetry operations do not close as assumed."
    )

    return Stars{T}(irreducible_directions, members, usable)
end

end
