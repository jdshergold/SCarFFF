# This module builds AO basis density matrices from the TD-DFT quantities that PySCF gives us:
# the X and Y amplitudes and the occupied and virtual MO coefficients.

module DensityMatrices

using LinearAlgebra

export build_transition_matrix, build_ground_state_matrix, build_difference_matrix, to_real_density

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

function build_difference_matrix(
        X::AbstractMatrix,
        Y::AbstractMatrix,
        mo_coeff_occ::AbstractMatrix,
        mo_coeff_vir::AbstractMatrix,
    )
    """
    Build the AO basis excited-minus-ground difference density matrix for one excited state,

        ΔT^(s) = C_v (Xᵀ X* + Y† Y) C_v† − C_o (X* Xᵀ + Y Y†) C_o†,

    which reduces to C_v (XᵀX + YᵀY) C_v^T − C_o (XXᵀ + YYᵀ) C_o^T for real coefficients and
    amplitudes.

    # Arguments:
    - X::AbstractMatrix: The excitation amplitudes, with shape (n_occupied, n_virtual).
    - Y::AbstractMatrix: The de-excitation amplitudes, with the same shape.
    - mo_coeff_occ::AbstractMatrix: The occupied MO coefficients C_o, with shape (n_ao, n_occupied).
    - mo_coeff_vir::AbstractMatrix: The virtual MO coefficients C_v, with shape (n_ao, n_virtual).

    # Returns:
    - The AO basis difference density matrix, with shape (n_ao, n_ao).
    """
    virtual_block = transpose(X) * conj(X) + adjoint(Y) * Y
    occupied_block = conj(X) * transpose(X) + Y * adjoint(Y)

    return mo_coeff_vir * virtual_block * adjoint(mo_coeff_vir) - mo_coeff_occ * occupied_block * adjoint(mo_coeff_occ)
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
