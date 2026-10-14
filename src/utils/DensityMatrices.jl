# This module builds AO basis density matrices from the TD-DFT quantities that PySCF gives us:
# the X and Y amplitudes and the occupied and virtual MO coefficients.

module DensityMatrices

using LinearAlgebra

export build_transition_matrix, build_ground_state_matrix, build_pair_difference_matrix,
       to_real_density, TransitionPairs, build_transition_pairs

function build_transition_matrix(
        X::AbstractMatrix,
        Y::AbstractMatrix,
        mo_coeff_occ::AbstractMatrix,
        mo_coeff_vir::AbstractMatrix,
    )
    """
    Build the AO basis transition density matrix for one excited state,

        T^(s) = C_o X* C_v† + C_v Y† C_o†.

    Written in the general form, with the conjugates and adjoints kept in place. For real MO
    coefficients and real X and Y it reduces to the familiar C_o (X + Y) C_v^T, because the second
    term is then the transpose of C_o Y C_v^T and the density is only ever used contracted with
    real AOs, where that transpose makes no difference.

    # Arguments:
    - X::AbstractMatrix: The excitation amplitudes, with shape (n_occupied, n_virtual).
    - Y::AbstractMatrix: The de-excitation amplitudes, with the same shape.
    - mo_coeff_occ::AbstractMatrix: The occupied MO coefficients C_o, with shape (n_ao, n_occupied).
    - mo_coeff_vir::AbstractMatrix: The virtual MO coefficients C_v, with shape (n_ao, n_virtual).

    # Returns:
    - The AO basis transition density matrix, with shape (n_ao, n_ao).
    """
    return mo_coeff_occ * conj(X) * adjoint(mo_coeff_vir) + mo_coeff_vir * adjoint(Y) * adjoint(mo_coeff_occ)
end

function build_ground_state_matrix(mo_coeff_occ::AbstractMatrix)
    """
    Build the AO basis ground state density matrix,

        T^(g) = C_o C_o†,

    which reduces to C_o C_o^T for real coefficients.

    # Arguments:
    - mo_coeff_occ::AbstractMatrix: The occupied MO coefficients C_o, with shape (n_ao, n_occupied).

    # Returns:
    - The AO basis ground state density matrix, with shape (n_ao, n_ao).
    """
    return mo_coeff_occ * adjoint(mo_coeff_occ)
end

function build_pair_difference_matrix(
        X_s::AbstractMatrix,
        Y_s::AbstractMatrix,
        X_t::AbstractMatrix,
        Y_t::AbstractMatrix,
        mo_coeff_occ::AbstractMatrix,
        mo_coeff_vir::AbstractMatrix,
    )
    """
    Build the AO basis difference density matrix, Δ_{st}, that contributes to the D term of the Hamiltonian,

        Δ_{st} ↔ C_v B^(ts) C_v† − C_o A^(ts) C_o†,
        A^(ts) = X^(s)* X^(t)ᵀ + Y^(t) Y^(s)†,
        B^(ts) = X^(t)ᵀ X^(s)* + Y^(s)† Y^(t).

    This is T^(ts), the transition density matrix from excited state t to excited state s, with
    δ_{st} times the ground state density subtracted.

    # Arguments:
    - X_s::AbstractMatrix: The excitation amplitudes of the row state, with shape (n_occupied, n_virtual).
    - Y_s::AbstractMatrix: The de-excitation amplitudes of the row state, with the same shape.
    - X_t::AbstractMatrix: The excitation amplitudes of the column state.
    - Y_t::AbstractMatrix: The de-excitation amplitudes of the column state.
    - mo_coeff_occ::AbstractMatrix: The occupied MO coefficients C_o, with shape (n_ao, n_occupied).
    - mo_coeff_vir::AbstractMatrix: The virtual MO coefficients C_v, with shape (n_ao, n_virtual).

    # Returns:
    - The AO basis density matrix of Δ_{st}, with shape (n_ao, n_ao).
    """
    virtual_block = transpose(X_t) * conj(X_s) + adjoint(Y_s) * Y_t
    occupied_block = conj(X_s) * transpose(X_t) + Y_t * adjoint(Y_s)

    return mo_coeff_vir * virtual_block * adjoint(mo_coeff_vir) - mo_coeff_occ * occupied_block * adjoint(mo_coeff_occ)
end

struct TransitionPairs
    """
    The (s, t) transition pairs that Δ_{st} is built for, and the map back to a single pair index.

    For real MO coefficients, amplitudes and AOs, Δ_{st} and Δ_{ts} give the same density, so we
    only need to store s <= t. We use the key:

        pair(s, t) = div(t * (t - 1), 2) + s.

    # Fields:
    - transitions_of_pair::Vector{NTuple{2, Int}}: The (s, t) of each pair index, with s <= t.
    - pair_of_transitions::Matrix{Int}: The pair index of each (s, t), symmetric in its arguments.
    """
    transitions_of_pair::Vector{NTuple{2, Int}}
    pair_of_transitions::Matrix{Int}
end

function build_transition_pairs(n_transitions::Int)::TransitionPairs
    """
    Build the (s, t) pair tables for a given number of transitions.

    # Arguments:
    - n_transitions::Int: The number of transitions, counted as positions in the requested list.

    # Returns:
    - TransitionPairs: The forward and reverse pair maps.
    """

    n_transitions >= 1 || error("There must be at least one transition, got $(n_transitions).")

    n_pairs = div(n_transitions * (n_transitions + 1), 2)
    transitions_of_pair = Vector{NTuple{2, Int}}(undef, n_pairs)
    pair_of_transitions = Matrix{Int}(undef, n_transitions, n_transitions)

    for t in 1:n_transitions, s in 1:t
        pair = div(t * (t - 1), 2) + s
        transitions_of_pair[pair] = (s, t)
        pair_of_transitions[s, t] = pair
        pair_of_transitions[t, s] = pair
    end

    return TransitionPairs(transitions_of_pair, pair_of_transitions)
end

function to_real_density(density::AbstractMatrix, ::Type{T}; tolerance = 1.0e-8, label::String = "density matrix") where {T<:AbstractFloat}
    """
    Convert a density matrix to real, after checking that it actually is real to within tolerance.

    The density matrices above are built in the general complex form, but everything downstream (the
    R tensor, the Cartesian contraction and the FFT integrand) is written for real densities. For our
    RKS calculations the MO coefficients and amplitudes are real, so the imaginary part is round-off,
    and this is where that assumption is stated and enforced rather than left implicit.

    # Arguments:
    - density::AbstractMatrix: The complex density matrix.
    - T::Type: The floating point type to convert to.
    - tolerance: The largest acceptable imaginary part, relative to the largest real part.
    - label::String: What to call the matrix in the error message.

    # Returns:
    - Matrix{T}: The real part of the density matrix.
    """

    real_scale = maximum(abs, real(density))
    imaginary_scale = maximum(abs, imag(density))

    if imaginary_scale > tolerance * max(real_scale, one(real_scale))
        error(
            "The $(label) has a significant imaginary part (largest |imag| = $(imaginary_scale) " *
            "against largest |real| = $(real_scale)). The form factor contraction paths assume real " *
            "densities, which holds for real MO coefficients and amplitudes."
        )
    end

    return Matrix{T}(real(density))
end

end
