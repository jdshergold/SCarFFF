# This script contains functions to construct the f_sq_lm tensor for spherical form factors.

module ConstructFSquaredTensor

using HDF5
using Base.Threads

include("../../utils/BinEncoding.jl")
using .BinEncoding: decode_bins
using ...SparseTensors: SparseGauntArray, lm_key
using ...FastPowers: fast_neg1_pow
using ...ThreadChunks: chunk_count, chunk_range

export construct_f_sq_lm_tensor

const INV_SQRT_2 = 1 / sqrt(2)

@inline function load_gaunt_array(path::String, ::Type{T}) where {T<:AbstractFloat}
    """
    Load a SparseGauntArray from an HDF5 file.

    # Arguments:
    - path::String: Path to the HDF5 file containing the Gaunt coefficients.
    - T::Type: The floating point type to use for the coefficients.

    # Returns:
    - SparseGauntArray{T}: The loaded Gaunt coefficient array.
    """
    h5open(path, "r") do io
        l1 = Vector{Int}(read(io, "l1"))
        m1 = Vector{Int}(read(io, "m1"))
        l2 = Vector{Int}(read(io, "l2"))
        l3 = Vector{Int}(read(io, "l3"))
        m3 = Vector{Int}(read(io, "m3"))
        Gaunt_values = Vector{T}(read(io, "Gaunt_values"))
        l1_m1_bins = decode_bins(Vector{Int}(read(io, "l1_m1_bins/data")), Vector{Int}(read(io, "l1_m1_bins/offsets")))
        return SparseGauntArray(l1, m1, l2, l3, m3, Gaunt_values, l1_m1_bins)
    end
end

@inline function U(μ::Int, m::Int, ::Type{T})::Complex{T} where {T<:AbstractFloat}
    """
    Compute U_{μm}, the complex-to-real spherical harmonic transformation matrix, defined as:

                    (1/√2) * (δ_{m,-μ} - iδ_{m,μ}),      if μ < 0,
        U_{μm} = {  δ_{m,0},                             if μ = 0,
                    ((-1)^m/√2) * (δ_{m,μ} + iδ_{m,-μ}), if μ > 0,

    where δ_{ab} is the Kronecker delta.

    # Arguments:
    - μ::Int: Magnetic quantum number of the complex spherical harmonic.
    - m::Int: Magnetic quantum number of the real spherical harmonic.
    - T::Type: The floating point type to use.

    # Returns:
    - Complex{T}: The transformation matrix element U_{μm}.
    """
    inv_sqrt_2 = T(INV_SQRT_2)
    if μ < 0
        return inv_sqrt_2 * Complex{T}(Int(m == -μ), -Int(m == μ))
    elseif μ == 0
        return Complex{T}(Int(m == 0))
    else
        return (fast_neg1_pow(m, T) * inv_sqrt_2) * Complex{T}(Int(m == μ), Int(m == -μ))
    end
end

function construct_f_sq_lm_tensor(
        R_tensor::Array{Complex{T}, 3},
        gaunt_f_sq_lm_path::String,
    )::Array{T, 3} where {T<:AbstractFloat}
    """
    Construct the f_{ℓm}^2(q) tensor from the R tensor, defined by:

        f_{ℓm}^2(q) = ∑_{ℓ1,m1} ∑_{ℓ2,m2} (-1)^{m2} R_{ℓ1,m1}(q) R*_{ℓ2,m2}(q) 𝒢_{ℓ1 m1,ℓ2 -m2}^{ℓμ} U_{μm},

    where μ = m1 - m2, 𝒢 is the Gaunt coefficient, and U_{μm} is the complex-to-real spherical
    harmonic transformation matrix. The resulting f_{ℓm}^2 are the real spherical harmonic coefficients
    of |f_s(q)|^2, and are real by construction. Any remaining imaginary part is
    numerical noise and is discarded before returning.

    The two R coefficients have indices (ℓ₁,m₁) and (ℓ₂,m₂). Forming |f|² conjugates
    the second harmonic, with conj(Y_ℓ₂^{m₂}) = (-1)^{m₂} Y_ℓ₂^{-m₂}. Thus the Gaunt
    coefficient couples (ℓ₁,m₁) and (ℓ₂,-m₂) to the output (ℓ,μ), where μ = m₁ - m₂.
    Its stored fields (l1,m1,l2,l3,m3) are (ℓ₁,m₁,ℓ₂,ℓ,μ). The code's m2 refers
    to the second R coefficient, so the Gaunt's second magnetic index is -m2.

    # Arguments:
    - R_tensor::Array{Complex{T}, 3}: The R tensor with shape (n_transitions, n_q, n_keys), keyed by key(ℓ, m) = ℓ^2 + (ℓ + m) + 1.
    - gaunt_f_sq_lm_path::String: Path to the precomputed Gaunt coefficients.

    # Returns:
    - f_sq_lm::Array{T, 3}: The f_{ℓm}^2 tensor with shape (n_transitions, n_q, n_keys),
      keyed identically to the R tensor.
    """

    # Load the Gaunt array.
    gaunt_array = load_gaunt_array(gaunt_f_sq_lm_path, T)
    gaunt_coeffs = gaunt_array.Gaunt_values

    # Get dimensions.
    n_transitions, n_q, n_keys = size(R_tensor)
    n_gaunt = length(gaunt_coeffs)

    # Allocate the output tensor.
    f_sq_lm = zeros(Complex{T}, n_transitions, n_q, n_keys)

    # Preallocate one accumulation buffer per task, keyed by chunk rather than by threadid so that
    # each buffer has exactly one writer. These are reused across transitions.
    n_chunks = chunk_count(n_gaunt, nthreads())
    f_sq_lm_local_pool = [zeros(Complex{T}, n_q, n_keys) for _ in 1:n_chunks]

    for transition_idx in 1:n_transitions
        # Zero out the per-task buffers for this transition.
        @inbounds for chunk in 1:n_chunks
            fill!(f_sq_lm_local_pool[chunk], zero(Complex{T}))
        end

        # Extract the R tensor slice for this transition. Use a view to avoid allocations.
        R = @view R_tensor[transition_idx, :, :]

        @sync for chunk in 1:n_chunks
            Threads.@spawn begin
                f_sq_lm_local = f_sq_lm_local_pool[chunk]

                for gaunt_idx in chunk_range(chunk, n_chunks, n_gaunt)
                    # Read the two amplitude angular momenta and the output (ℓ, μ).
                    l1 = gaunt_array.l1[gaunt_idx]
                    m1 = gaunt_array.m1[gaunt_idx]
                    l2 = gaunt_array.l2[gaunt_idx]
                    l = gaunt_array.l3[gaunt_idx]
                    mu = gaunt_array.m3[gaunt_idx]
                    # m2 is the second R coefficient's magnetic index, before conjugation.
                    m2 = m1 - mu

                    gaunt_val = gaunt_coeffs[gaunt_idx]

                    # Compute the R tensor keys.
                    key1 = l1 * l1 + (l1 + m1) + 1 # key(ℓ1, m1)
                    key2 = l2 * l2 + (l2 + m2) + 1 # key(ℓ2, m2)

                    # The (-1)^{m2} sign factor from conjugating the second spherical harmonic.
                    sign_m2 = fast_neg1_pow(m2, T)

                    # U_{μm} is non-zero only for m = ±|μ| (or just 0 when μ = 0).
                    # Loop over the at most two non-zero m values.
                    if mu == 0
                        # U_{0, 0} = 1; output real harmonic index is 0.
                        key_out = l * l + l + 1
                        combined = Complex{T}(sign_m2 * gaunt_val)
                        @inbounds for q_idx in 1:n_q
                            f_sq_lm_local[q_idx, key_out] += combined * R[q_idx, key1] * conj(R[q_idx, key2])
                        end
                    else
                        # U_{μ, μ} and U_{μ, -μ} are the two non-zero entries.
                        for m in (mu, -mu)
                            u_val = U(mu, m, T)
                            combined = sign_m2 * gaunt_val * u_val
                            key_out = l * l + (l + m) + 1
                            @inbounds for q_idx in 1:n_q
                                f_sq_lm_local[q_idx, key_out] += combined * R[q_idx, key1] * conj(R[q_idx, key2])
                            end
                        end
                    end
                end
            end
        end

        # Accumulate the per-task results into the global f_sq_lm tensor.
        @inbounds for f_sq_lm_per_chunk in f_sq_lm_local_pool
            f_sq_lm[transition_idx, :, :] .+= f_sq_lm_per_chunk
        end
    end

    return real.(f_sq_lm)
end

end
