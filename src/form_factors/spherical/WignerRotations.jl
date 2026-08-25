# This module builds the Wigner D matrix blocks that rotate spherical harmonic coefficients.

module WignerRotations

using Quaternionic
using VectorSpaceDarkMatter

const VSDM = VectorSpaceDarkMatter

# SphericalFunctions.jl stores its Wigner D matrices transposed and/or conjugated relative to its own
# documentation, so we follow the same load-time probe VSDM uses rather than hardcoding an assumption.
const D_NEEDS_TRANSPOSE = VSDM.do_transpose
const D_NEEDS_CONJUGATE = VSDM.do_conjugate

export WignerDBuffers, wigner_d_blocks, wigner_d_blocks!

struct WignerDBuffers{T<:AbstractFloat, B}
    """
    Reusable storage for the Wigner D matrices up to one value of l_max.

    # Fields:
    - buffer::B: The working arrays used by VectorSpaceDarkMatter to construct all D coefficients.
    - blocks::Vector{Matrix{Complex{T}}}: The square D matrix for each l.
    """
    buffer::B
    blocks::Vector{Matrix{Complex{T}}}
end

function WignerDBuffers(::Type{T}, l_max::Int) where {T<:AbstractFloat}
    """
    Allocate the Wigner D storage used repeatedly by one thread.

    # Arguments:
    - T::Type{<:AbstractFloat}: Floating-point precision of the returned D matrices.
    - l_max::Int: The largest angular mode required.

    # Returns:
    - WignerDBuffers: Reusable VectorSpaceDarkMatter storage and one matrix per l.
    """

    buffer = VSDM.D_prep(l_max)
    blocks = [Matrix{Complex{T}}(undef, 2 * l + 1, 2 * l + 1) for l in 0:l_max]
    return WignerDBuffers{T, typeof(buffer)}(buffer, blocks)
end

function wigner_d_blocks!(
        buffers::WignerDBuffers{T},
        rotation::Quaternionic.Rotor{T},
    )::Vector{Matrix{Complex{T}}} where {T<:AbstractFloat}
    """
    Fill reusable Wigner D blocks for one rotation,

        Y_l^m(R̃^{-1} q̂) = Σ_μ D_{μm}^(l)(R̃) Y_l^μ(q̂).

    # Arguments:
    - buffers::WignerDBuffers{T}: Per-thread storage sized to the required l_max.
    - rotation::Quaternionic.Rotor{T}: The proper rotation R̃.

    # Returns:
    - Vector{Matrix{Complex{T}}}: The buffer's blocks, indexed [μ + l + 1, m + l + 1].
    """

    VSDM.D_matrices!(buffers.buffer, rotation)
    D_values = buffers.buffer[1]

    @inbounds for l in 0:(length(buffers.blocks) - 1)
        block = buffers.blocks[l + 1]
        width = 2 * l + 1
        block_start = VSDM.WignerDindex(l, -l, -l)
        for column in 1:width, row in 1:width
            # VSDM's flat storage is reshaped column-major before its transpose and
            # conjugation are applied. Do both operations while copying into the reused block.
            # Do it this way (1D) to avoid allocating a 2D intermediate, as this is done many many times.
            raw_offset = D_NEEDS_TRANSPOSE ? (row - 1) * width + column - 1 :
                                              (column - 1) * width + row - 1
            value = Complex{T}(D_values[block_start + raw_offset])
            block[row, column] = D_NEEDS_CONJUGATE ? conj(value) : value
        end
    end

    return buffers.blocks
end

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

    buffers = WignerDBuffers(T, l_max)
    return wigner_d_blocks!(buffers, rotation)
end

end
