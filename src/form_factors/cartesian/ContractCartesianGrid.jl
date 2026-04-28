# This module contracts the V tensors to form the full 3D Cartesian form factor grid.

module ContractCartesianGrid

using Tullio
using Base.Threads

export contract_cartesian_grid

# The √2 factor is for spin-degeneracy.
const prefactor = sqrt(2) * (2*π)^(3/2)

function contract_cartesian_grid(
    V_x::Array{Complex{T}, 2},
    V_y::Array{Complex{T}, 2},
    V_z::Array{Complex{T}, 2},
    nonzero_pairs::Vector{Tuple{Int, Int}},
    M_ij::Array{T, 2},
    transition_matrices::Vector{Matrix{T}},
    cartesian_term_to_orbital::Vector{Int};
    threshold::T = zero(T)
)::Array{Complex{T}, 4} where {T<:AbstractFloat}
    """
    Contract the separable 1D V tensors to form the full 4D form factor grid for all transitions:

        f_s(q) = (2π)^{3/2} Σ_p TDM_prefactor[t, p] * V_x[q_x, p] * V_y[q_y, p] * V_z[q_z, p],

    where t denotes the transition index, and V_x is weighted by M_ij. For each transition,
    pairs are discarded if |T_ij| = |M_ij * TDM| < max{|T_ij|} * threshold.

    # Arguments:
    - V_x::Array{Complex{T},2}: The M_ij-weighted V tensor for the x-direction, with shape [n_q_x, n_pairs].
    - V_y::Array{Complex{T},2}: The V tensor for the y-direction, with shape [n_q_y, n_pairs].
    - V_z::Array{Complex{T},2}: The V tensor for the z-direction, with shape [n_q_z, n_pairs].
    - nonzero_pairs::Vector{Tuple{Int, Int}}: The list of (i, j) pairs that passed the M_ij threshold.
    - M_ij::Array{T,2}: The geometry-only pair coefficients.
    - transition_matrices::Vector{Matrix{T}}: The transition density matrices for all requested transitions.
    - cartesian_term_to_orbital::Vector{Int}: Mapping from Cartesian term index to orbital index.
    - threshold::T: A threshold value below which pairs satisfying (|T_ij/T_max| < threshold) will be discarded per transition (default: 0.0).

    # Returns:
    - form_factor::Array{Complex{T},4}: The form factor on the 3D Cartesian grid for all transitions, with shape [n_transitions, n_q_x, n_q_y, n_q_z].
    """

    # Extract the relevant dimensions for the output array.
    n_transitions = length(transition_matrices)
    n_pairs = length(nonzero_pairs)
    n_q_x = size(V_x, 1)
    n_q_y = size(V_y, 1)
    n_q_z = size(V_z, 1)

    # Allocate the output array.
    form_factor = Array{Complex{T}}(undef, n_transitions, n_q_x, n_q_y, n_q_z)

    # Process each transition separately to enable per-transition filtering.
    for transition_idx in 1:n_transitions
        TDM = transition_matrices[transition_idx]

        # Precompute TDM prefactors for this transition.
        tdm_prefactors = Vector{T}(undef, n_pairs)
        @threads for pair_idx in 1:n_pairs
            pair_i, pair_j = nonzero_pairs[pair_idx]
            orbital_i = cartesian_term_to_orbital[pair_i]
            orbital_j = cartesian_term_to_orbital[pair_j]

            if pair_i == pair_j
                # For diagonal entries there is only one contribution.
                tdm_prefactors[pair_idx] = TDM[orbital_i, orbital_j]
            else
                # Otherwise we add the ij and ji contributions.
                tdm_prefactors[pair_idx] = TDM[orbital_i, orbital_j] + TDM[orbital_j, orbital_i]
            end
        end

        # Compute T_ij = M_ij * TDM_prefactor for each pair.
        T_ij_vals = Vector{T}(undef, n_pairs)
        @threads for pair_idx in 1:n_pairs
            pair_i, pair_j = nonzero_pairs[pair_idx]
            T_ij_vals[pair_idx] = M_ij[pair_i, pair_j] * tdm_prefactors[pair_idx]
        end

        # Identify the pairs that survive the threshold.
        T_max = maximum(abs.(T_ij_vals))
        pair_threshold = threshold * T_max
        surviving_pairs = Int[]
        for pair_idx in 1:n_pairs
            if abs(T_ij_vals[pair_idx]) > pair_threshold
                push!(surviving_pairs, pair_idx)
            end
        end

        n_surviving = length(surviving_pairs)

        # Create a view into the form factor array. This stops Tullio from performing an extra loop over transitions.
        form_factor_transition = view(form_factor, transition_idx, :, :, :)

        # If no pairs survive, skip this transition.
        if n_surviving == 0
            fill!(form_factor_transition, zero(Complex{T}))
            continue
        end

        # Perform the 1D contraction over surviving pairs using Tullio with threading.
        @tullio threads=true form_factor_transition[qx, qy, qz] = begin
            idx = surviving_pairs[p]
            tdm_prefactors[idx] * V_x[qx, idx] * V_y[qy, idx] * V_z[qz, idx]
        end
    end

    return T(prefactor) .* form_factor
end

end
