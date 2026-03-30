# This module constructs the 1D V tensors for the Cartesian form factor.

module ConstructVTensors

using Base.Threads

using ..ReadBasisSet: MoleculeData
using ..ProbabilistsHermite: fill_hermite_vector!

export construct_V_tensors

function construct_V_tensors(
    mol::MoleculeData{T},
    M_ij::Array{T, 2},
    sigma_ij::Array{T, 2},
    R_ij::Array{T, 3},
    b_A::Array{T, 3},
    b_B::Array{T, 3},
    b_C::Array{T, 3},
    q_x_vals::Vector{T},
    q_y_vals::Vector{T},
    q_z_vals::Vector{T};
    threshold::T = 0.0,
)::Tuple{Array{Complex{T}, 2}, Array{Complex{T}, 2}, Array{Complex{T}, 2}, Vector{Tuple{Int, Int}}} where {T<:AbstractFloat}
    """
    Construct the transition-independent 1D V tensors for all Cartesian pairs, defined by:

        V_{ij}(q_x) = exp(-q_x^2 σ_ij^2 / 2) exp(iq X_ij) Σ_A b_ij^A (iσ_ij)^A He_A(q_x σ_ij),

    and similarly for the y and z components.

    # Arguments:
    - mol::MoleculeData{T}: The molecular data structure.
    - M_ij::Array{T,2}: The molecule-specific pair coefficients.
    - sigma_ij::Array{T,2}: The combined Gaussian widths.
    - R_ij::Array{T,3}: The weighted position vectors.
    - b_A::Array{T,3}: The b^A coefficients for the x-direction.
    - b_B::Array{T,3}: The b^B coefficients for the y-direction.
    - b_C::Array{T,3}: The b^C coefficients for the z-direction.
    - q_x_vals::Vector{T}: The q-values along the x-axis.
    - q_y_vals::Vector{T}: The q-values along the y-axis.
    - q_z_vals::Vector{T}: The q-values along the z-axis.
    - threshold::T: A threshold value below which pairs satisfying (|M/M_max| < threshold) will be discarded (default: 0.0, so no discarding).

    # Returns:
    - V_x::Array{Complex{T},2}: The V tensor for the x-direction, with shape [n_qx, n_pairs].
    - V_y::Array{Complex{T},2}: The V tensor for the y-direction, with shape [n_qy, n_pairs].
    - V_z::Array{Complex{T},2}: The V tensor for the z-direction, with shape [n_qz, n_pairs].
    - nonzero_pairs::Vector{Tuple{Int, Int}}: The list of (i, j) pairs that passed the threshold.
    """

    n_cartesian_terms = mol.n_cartesian_terms
    n_q_x = length(q_x_vals)
    n_q_y = length(q_y_vals)
    n_q_z = length(q_z_vals)

    # Compute the threshold using the maximum M_ij value.
    M_max = maximum(abs.(M_ij))
    pair_threshold = threshold * M_max

    # First we store the indices of any pairs above the threshold.
    nonzero_pairs = Tuple{Int, Int}[]
    for i in 1:n_cartesian_terms
        for j in i:n_cartesian_terms
            if abs(M_ij[i, j]) > pair_threshold
                push!(nonzero_pairs, (i, j))
            end
        end
    end


    # Now we allocate arrays for the V tensors.
    n_pairs = length(nonzero_pairs)
    V_x = Array{Complex{T}}(undef, n_q_x, n_pairs)
    V_y = Array{Complex{T}}(undef, n_q_y, n_pairs)
    V_z = Array{Complex{T}}(undef, n_q_z, n_pairs)

    # Pre-allocate Hermite polynomial buffers for each thread to avoid allocations in the inner loop.
    # As we go up to at most i-orbitals, max Cartesian power is 12, so we need buffers of size 13.
    He_buffers = [Vector{T}(undef, 13) for _ in 1:nthreads()]

    # Now we loop over the non-thresholded pairs and compute the V tensors.
    @threads for pair_idx in 1:n_pairs
        i, j = nonzero_pairs[pair_idx]

        # Get the Hermite buffer for this thread.
        He_buffer = He_buffers[threadid()]

        # Get the M_ij coefficient for weighting.
        M_ij_val = M_ij[i, j]

        # Extract the Cartesian powers.
        a_i = mol.cartesian_a[i]
        b_i = mol.cartesian_b[i]
        c_i = mol.cartesian_c[i]
        a_j = mol.cartesian_a[j]
        b_j = mol.cartesian_b[j]
        c_j = mol.cartesian_c[j]

        # Extract the pair-specific quantities.
        sigma_ij_val = sigma_ij[i, j]
        X_ij = R_ij[i, j, 1]
        Y_ij = R_ij[i, j, 2]
        Z_ij = R_ij[i, j, 3]

        # Determine the maximum orders for this pair.
        max_A = a_i + a_j
        max_B = b_i + b_j
        max_C = c_i + c_j

        # Precompute common quantities.
        sigma_ij_sq = sigma_ij_val * sigma_ij_val

        # First, we compute V_x for this pair.
        for (idx, q_x) in enumerate(q_x_vals)
            gaussian = exp(-T(0.5) * q_x * q_x * sigma_ij_sq)
            phase = exp(im * q_x * X_ij)

            # Evaluate the Hermite polynomials in-place using the thread-local buffer.
            fill_hermite_vector!(He_buffer, max_A, q_x * sigma_ij_val)

            hermite_sum = zero(Complex{T})
            i_sigma_power = one(Complex{T})

            for A in 0:max_A
                hermite_sum += b_A[i, j, A + 1] * i_sigma_power * He_buffer[A + 1]
                i_sigma_power *= im * sigma_ij_val
            end

            # Weight V_x by M_ij.
            V_x[idx, pair_idx] = M_ij_val * gaussian * phase * hermite_sum
        end

        # Then V_y.
        for (idx, q_y) in enumerate(q_y_vals)
            gaussian = exp(-T(0.5) * q_y * q_y * sigma_ij_sq)
            phase = exp(im * q_y * Y_ij)

            # Evaluate the Hermite polynomials in-place using the thread-local buffer.
            fill_hermite_vector!(He_buffer, max_B, q_y * sigma_ij_val)

            hermite_sum = zero(Complex{T})
            i_sigma_power = one(Complex{T})

            for B in 0:max_B
                hermite_sum += b_B[i, j, B + 1] * i_sigma_power * He_buffer[B + 1]
                i_sigma_power *= im * sigma_ij_val
            end

            V_y[idx, pair_idx] = gaussian * phase * hermite_sum
        end

        # Finally, V_z.
        for (idx, q_z) in enumerate(q_z_vals)
            gaussian = exp(-T(0.5) * q_z * q_z * sigma_ij_sq)
            phase = exp(im * q_z * Z_ij)

            # Evaluate the Hermite polynomials in-place using the thread-local buffer.
            fill_hermite_vector!(He_buffer, max_C, q_z * sigma_ij_val)

            hermite_sum = zero(Complex{T})
            i_sigma_power = one(Complex{T})

            for C in 0:max_C
                hermite_sum += b_C[i, j, C + 1] * i_sigma_power * He_buffer[C + 1]
                i_sigma_power *= im * sigma_ij_val
            end

            V_z[idx, pair_idx] = gaussian * phase * hermite_sum
        end
    end

    return V_x, V_y, V_z, nonzero_pairs
end

end