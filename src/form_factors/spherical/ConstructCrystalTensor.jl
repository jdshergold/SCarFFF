# This script contains functions to construct crystal FLM tensors from conformer FLM tensors.

module ConstructCrystalTensor

using Quaternionic
using VectorSpaceDarkMatter
import LinearAlgebra: mul!
import Logging: NullLogger, with_logger

const VSDM = VectorSpaceDarkMatter

export construct_crystal_f_lm_tensors

function get_G_blocks(rotation::Quaternionic.Rotor{T}, l_max::Int)::Vector{Matrix{T}} where {T<:AbstractFloat}
    """
    Construct the real spherical harmonic rotation blocks for a rotor.

    # Arguments:
    - rotation::Quaternionic.Rotor{T}: The proper rotation as a unit quaternion rotor.
    - l_max::Int: The maximum angular momentum mode.

    # Returns:
    - G_blocks::Vector{Matrix{T}}: The G matrix block for each angular momentum mode.
    """

    # Compute the Wigner D matrices and the real spherical harmonic rotation matrix G.
    D = VSDM.D_prep(l_max)
    G = zeros(Float64, VSDM.WignerDsize(l_max))
    null_logger = NullLogger()

    with_logger(null_logger) do
        VSDM.D_matrices!(D, rotation)
        VSDM.G_matrices!(G, D)
    end

    # Extract the Nm * Nm block for each l from the flat G vector.
    G_blocks = Vector{Matrix{T}}(undef, l_max + 1)
    for l in 0:l_max
        # Get the start and end indices of each block.
        block_start = VSDM.WignerDindex(l, -l, -l)
        block_stop = VSDM.WignerDindex(l, l, l)
        # Turn this block into an Nm * Nm matrix.
        G_blocks[l + 1] = T.(reshape(copy(@view G[block_start:block_stop]), 2 * l + 1, 2 * l + 1))
    end

    return G_blocks
end

function accumulate_rotated_f_lm!(
        output_f_lm::Array{T, 3},
        input_f_lm::Array{T, 3},
        proper_rotation::Quaternionic.Rotor{T},
        det_rotation::T,
    ) where {T<:AbstractFloat}
    """
    Rotate a conformer FLM tensor into the crystal frame and accumulate it into the output tensor.

    # Arguments:
    - output_f_lm::Array{T, 3}: The output tensor to accumulate into.
    - input_f_lm::Array{T, 3}: The input conformer FLM tensor.
    - proper_rotation::Quaternionic.Rotor{T}: The proper rotation as a unit quaternion rotor.
    - det_rotation::T: The determinant of the original orthogonal matrix R.

    # Returns:
    - None.
    """

    l_max = Int(sqrt(size(input_f_lm, 3)) - 1) # (l+1)^2 -> l.
    G_blocks = get_G_blocks(proper_rotation, l_max)

    # Preallocate a buffer for the rotated block, reused across all (l, transition, q).
    rotated_block = zeros(T, 2 * l_max + 1)

    # Loop over angular momentum modes and apply the rotation matrix to each.
    for l in 0:l_max
        key_start = l * l + 1
        key_stop = (l + 1) * (l + 1)
        # For improper rotations (det = -1), odd-l blocks pick up a sign flip.
        # Proper rotations do not. This can be neatly unified as det(R)^{\ell}.
        parity_l = det_rotation^l
        G_l = G_blocks[l + 1]
        # Create a "shortened" view into the big rotated block for this l.
        rotated_block_l = @view rotated_block[1:2 * l + 1]

        for transition_idx in axes(input_f_lm, 1)
            for q_idx in axes(input_f_lm, 2)
                # Create views into each of the input and output arrays.
                input_block = @view input_f_lm[transition_idx, q_idx, key_start:key_stop]
                output_block = @view output_f_lm[transition_idx, q_idx, key_start:key_stop]
                # Rotate in place as rot = G @ input, then accumulate.
                mul!(rotated_block_l, G_l, input_block)
                @. output_block += parity_l * rotated_block_l
            end
        end
    end
end

function construct_crystal_f_lm_tensors(
        conformer_labels::Vector{String},
        conformer_f_lm::Vector{Array{T, 3}},
        conformer_sets,
    )::Tuple{Vector{Array{T, 3}}, Array{T, 3}} where {T<:AbstractFloat}
    """
    Construct crystal FLM tensors for each conformer set, and the loose aggregate.

    # Arguments:
    - conformer_labels::Vector{String}: Labels for the conformers.
    - conformer_f_lm::Vector{Array{T, 3}}: The conformer FLM tensors.
    - conformer_sets: The conformer-set metadata.

    # Returns:
    - set_f_lm::Vector{Array{T, 3}}: The FLM tensor for each conformer set.
    - aggregate_f_lm::Array{T, 3}: The occupancy-weighted loose aggregate FLM tensor.
    """

    # Build a label -> index map so we can look up conformer FLM tensors by label.
    label_to_idx = Dict(label => idx for (idx, label) in enumerate(conformer_labels))

    # Initialise output tensors.
    set_f_lm = Vector{Array{T, 3}}(undef, length(conformer_sets))
    aggregate_f_lm = zeros(T, size(conformer_f_lm[1]))

    for (set_idx, conformer_set) in enumerate(conformer_sets)
        label = conformer_set.label
        occupancy = conformer_set.occupancy

        if !haskey(label_to_idx, label)
            error("No conformer FLM tensor found for conformer label $(label).")
        end

        # Accumulate each image of this conformer into the set tensor.
        base_f_lm = conformer_f_lm[label_to_idx[label]]
        set_tensor = zeros(T, size(base_f_lm))

        for (proper_rotation, det_rotation) in zip(conformer_set.proper_rotations, conformer_set.det_rotations)
            accumulate_rotated_f_lm!(set_tensor, base_f_lm, proper_rotation, det_rotation)
        end

        # Accumulate into the aggregate with the disorder group occupancy weight.
        set_f_lm[set_idx] = set_tensor
        aggregate_f_lm .+= occupancy .* set_tensor
    end

    return set_f_lm, aggregate_f_lm
end

end
