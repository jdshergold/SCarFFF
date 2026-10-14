# This module contracts the V tensors to form the full 3D Cartesian form factor grid using GPU matrix multiplications.

module ContractCartesianGridGPU

using CUDA
using Base.Threads

export contract_cartesian_grid_gpu

# The factor of 2 accounts for the two spin channels, since the transition matrices carry
# the per-spin-channel X_α and Y_α straight from PySCF.
const prefactor = 2 * (2*π)^(3/2)

function contract_cartesian_grid_gpu(
    V_x::Array{Complex{T}, 2},
    V_y::Array{Complex{T}, 2},
    V_z::Array{Complex{T}, 2},
    nonzero_pairs::Vector{Tuple{Int, Int}},
    M_ij::Array{T, 2},
    T_gs::Vector{Matrix{T}},
    cartesian_term_to_orbital::Vector{Int};
    threshold::T = zero(T)
)::Array{Complex{T}, 4} where {T<:AbstractFloat}
    """
    Contract the separable 1D V tensors to form the full 4D form factor grid for all transitions using GPU matrix multiplications:

        f_s(q) = 2 (2π)^{3/2} Σ_p density_pair_weight[t, p] * V_x[q_x, p] * V_y[q_y, p] * V_z[q_z, p],

    where t denotes the transition index, and V_x is weighted by M_ij. For each transition,
    pairs are discarded if |weighted_density_pair| = |M_ij * T_gs_state| < max{|weighted_density_pair|} * threshold.

    # Arguments:
    - V_x::Array{Complex{T},2}: The M_ij-weighted V tensor for the x-direction, with shape [n_q_x, n_pairs].
    - V_y::Array{Complex{T},2}: The V tensor for the y-direction, with shape [n_q_y, n_pairs].
    - V_z::Array{Complex{T},2}: The V tensor for the z-direction, with shape [n_q_z, n_pairs].
    - nonzero_pairs::Vector{Tuple{Int, Int}}: The list of (i, j) pairs that passed the M_ij threshold.
    - M_ij::Array{T,2}: The geometry-only pair coefficients.
    - T_gs::Vector{Matrix{T}}: The per-spin AO transition matrices for all requested transitions.
    - cartesian_term_to_orbital::Vector{Int}: Mapping from Cartesian term index to orbital index.
    - threshold::T: A threshold value below which pairs satisfying (|weighted_density_pair/max_weighted_density_pair| < threshold) will be discarded per transition (default: 0.0).

    # Returns:
    - form_factor::Array{Complex{T},4}: The form factor on the 3D Cartesian grid for all transitions, with shape [n_transitions, n_q_x, n_q_y, n_q_z].
    """

    # Extract the relevant dimensions for the output array.
    n_transitions = length(T_gs)
    n_pairs = length(nonzero_pairs)
    n_q_x = size(V_x, 1)
    n_q_y = size(V_y, 1)
    n_q_z = size(V_z, 1)

    # Allocate the output array.
    form_factor = Array{Complex{T}}(undef, n_transitions, n_q_x, n_q_y, n_q_z)

    # Process each transition separately to enable per-transition filtering.
    for transition_idx in 1:n_transitions
        T_gs_state = T_gs[transition_idx]

        # Precompute T_gs_state prefactors for this transition.
        density_pair_weights = Vector{T}(undef, n_pairs)
        @threads for pair_idx in 1:n_pairs
            pair_i, pair_j = nonzero_pairs[pair_idx]
            orbital_i = cartesian_term_to_orbital[pair_i]
            orbital_j = cartesian_term_to_orbital[pair_j]

            if pair_i == pair_j
                # For diagonal entries there is only one contribution.
                density_pair_weights[pair_idx] = T_gs_state[orbital_i, orbital_j]
            else
                # Otherwise we add the ij and ji contributions.
                density_pair_weights[pair_idx] = T_gs_state[orbital_i, orbital_j] + T_gs_state[orbital_j, orbital_i]
            end
        end

        # Compute weighted_density_pair = M_ij * density_pair_weight for each pair.
        weighted_density_pairs = Vector{T}(undef, n_pairs)
        @threads for pair_idx in 1:n_pairs
            pair_i, pair_j = nonzero_pairs[pair_idx]
            weighted_density_pairs[pair_idx] = M_ij[pair_i, pair_j] * density_pair_weights[pair_idx]
        end

        # Identify the pairs that survive the threshold.
        max_weighted_density_pair = maximum(abs.(weighted_density_pairs))
        pair_threshold = threshold * max_weighted_density_pair
        surviving_pairs = Int[]
        for pair_idx in 1:n_pairs
            if abs(weighted_density_pairs[pair_idx]) > pair_threshold
                push!(surviving_pairs, pair_idx)
            end
        end

        n_surviving = length(surviving_pairs)

        # Create a view into the form factor array.
        form_factor_transition = view(form_factor, transition_idx, :, :, :)

        # If no pairs survive, skip this transition.
        if n_surviving == 0
            fill!(form_factor_transition, zero(Complex{T}))
            continue
        end

        # Extract the surviving pair columns from V tensors and tdm prefactors.
        V_x_surviving = V_x[:, surviving_pairs]
        V_y_surviving = V_y[:, surviving_pairs]
        V_z_surviving = V_z[:, surviving_pairs]
        tdm_surviving = density_pair_weights[surviving_pairs]

        # Move V_y, V_z, and tdm to GPU once.
        V_y_gpu = CuArray(V_y_surviving)
        V_z_gpu = CuArray(V_z_surviving)
        tdm_gpu = CuArray(tdm_surviving)

        # Scale V_z by the T_gs_state prefactors.
        V_z_scaled_gpu = V_z_gpu .* reshape(tdm_gpu, (1, n_surviving))

        # Determine optimal qx_chunk_size based on available GPU memory.
        # Memory per chunk: qx_chunk × qy × n_surviving × sizeof(Complex{T}) for temp,
        #                 + qx_chunk × qy × n_qz × sizeof(Complex{T}) for result.
        available_memory = CUDA.available_memory()
        bytes_per_qx_slice = (n_q_y * n_surviving + n_q_y * n_q_z) * sizeof(Complex{T})

        # Target using ~25% of available memory for the chunk buffers.
        target_memory = T(0.25) * available_memory
        max_qx_chunk_size = max(1, floor(Int, target_memory / bytes_per_qx_slice))

        qx_chunk_size = min(max_qx_chunk_size, n_q_x)
        qx_chunk_size = max(1, qx_chunk_size)

        n_chunks = cld(n_q_x, qx_chunk_size)

        # Process V_x in chunks to avoid huge intermediate arrays.
        for chunk_idx in 1:n_chunks
            # Determine the qx range for this chunk.
            qx_start = (chunk_idx - 1) * qx_chunk_size + 1
            qx_end = min(chunk_idx * qx_chunk_size, n_q_x)
            chunk_n_qx = qx_end - qx_start + 1

            # Extract and transfer the chunk of V_x to GPU.
            V_x_chunk_cpu = V_x_surviving[qx_start:qx_end, :]
            V_x_chunk_gpu = CuArray(V_x_chunk_cpu)

            # Compute the outer product for this chunk: temp[qx_chunk, qy, p] = V_x[qx_chunk, p] * V_y[qy, p].
            V_x_reshaped = reshape(V_x_chunk_gpu, (chunk_n_qx, 1, n_surviving))
            V_y_reshaped = reshape(V_y_gpu, (1, n_q_y, n_surviving))
            temp_chunk_gpu = V_x_reshaped .* V_y_reshaped

            # Reshape temp to (chunk_n_qx * n_q_y, n_surviving) for matrix multiplication.
            temp_flat_gpu = reshape(temp_chunk_gpu, (chunk_n_qx * n_q_y, n_surviving))

            # Multiply with V_z_scaled^T: result = temp_flat * V_z_scaled^T.
            # This gives (chunk_n_qx * n_q_y, n_q_z).
            result_flat_gpu = temp_flat_gpu * transpose(V_z_scaled_gpu)

            # Reshape to (chunk_n_qx, n_q_y, n_q_z).
            result_chunk_gpu = reshape(result_flat_gpu, (chunk_n_qx, n_q_y, n_q_z))

            # Copy the result back to CPU and insert into the appropriate slice.
            form_factor_transition[qx_start:qx_end, :, :] .= Array(result_chunk_gpu)
        end
    end

    return T(prefactor) .* form_factor
end

end
