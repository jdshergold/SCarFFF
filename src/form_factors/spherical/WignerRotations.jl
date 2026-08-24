# This module builds the Wigner D matrix blocks that rotate spherical harmonic coefficients.

module WignerRotations

using Quaternionic
using VectorSpaceDarkMatter

const VSDM = VectorSpaceDarkMatter

# SphericalFunctions.jl stores its Wigner D matrices transposed and/or conjugated relative to its own
# documentation, so we follow the same load-time probe VSDM uses rather than hardcoding an assumption.
const D_NEEDS_TRANSPOSE = VSDM.do_transpose
const D_NEEDS_CONJUGATE = VSDM.do_conjugate

export wigner_d_blocks

function wigner_d_blocks(rotation::Quaternionic.Rotor{T}, l_max::Int)::Vector{Matrix{Complex{T}}} where {T<:AbstractFloat}
    """
    Build the Wigner D matrix blocks that rotate spherical harmonic coefficients, in the convention

        Y_l^m(R̃^{-1} q̂) = Σ_μ D^{(l)}_{μm}(R̃) Y_l^μ(q̂).

    VSDM stores its D coefficients flat, so we extract the (2l+1)x(2l+1) block for each l and apply
    whichever of transpose and conjugate the probe above calls for, which puts them in the [μ, m]
    orientation this convention needs.

    # Arguments:
    - rotation::Quaternionic.Rotor{T}: The proper rotation R̃ as a unit quaternion rotor.
    - l_max::Int: The maximum angular momentum mode.

    # Returns:
    - Vector{Matrix{Complex{T}}}: The D block for each l, indexed [μ + l + 1, m + l + 1].
    """

    D_buffer = VSDM.D_prep(l_max)
    VSDM.D_matrices!(D_buffer, rotation)
    D_values = D_buffer[1]

    blocks = Vector{Matrix{Complex{T}}}(undef, l_max + 1)
    for l in 0:l_max
        block_start = VSDM.WignerDindex(l, -l, -l)
        block_stop = VSDM.WignerDindex(l, l, l)
        block = reshape(collect(D_values[block_start:block_stop]), 2 * l + 1, 2 * l + 1)
        # Rotate if needed, as per VSDM.
        D_NEEDS_TRANSPOSE && (block = permutedims(block))
        D_NEEDS_CONJUGATE && (block = conj(block))
        blocks[l + 1] = Matrix{Complex{T}}(block)
    end

    return blocks
end

end
