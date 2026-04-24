# This script contains functions to construct the f_lm tensor for spherical form factors.

module ConstructFLMTensor

using HDF5
using Base.Threads

include("../../utils/BinEncoding.jl")
using .BinEncoding: decode_bins
using ...SparseTensors: SparseGauntArray, lambda_mu_key
using ...FastPowers: fast_neg1_pow

export construct_f_lm_tensor

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
        lambda = Vector{Int}(read(io, "lambda"))
        mu = Vector{Int}(read(io, "mu"))
        L = Vector{Int}(read(io, "L"))
        l = Vector{Int}(read(io, "l"))
        m = Vector{Int}(read(io, "m"))
        coeffs = Vector{T}(read(io, "coefficients"))
        lambda_mu_bins = decode_bins(Vector{Int}(read(io, "lambda_mu_bins/data")), Vector{Int}(read(io, "lambda_mu_bins/offsets")))
        return SparseGauntArray(lambda, mu, L, l, m, coeffs, lambda_mu_bins)
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

function construct_f_lm_tensor(
        R_tensor::Array{Complex{T}, 3},
        gaunt_flm_path::String,
    )::Array{T, 3} where {T<:AbstractFloat}
    """
    Construct the f_{ℓm}^2(q) tensor from the R tensor, defined by:

        f_{ℓm}^2(q) = ∑_{ℓ1,m1} ∑_{ℓ2,m2} (-1)^{m2} R_{ℓ1,m1}(q) R*_{ℓ2,m2}(q) G_{ℓ1,ℓ2,ℓ}^{m1,μ} U_{μm},

    where μ = m1 - m2, G is the Gaunt coefficient, and U_{μm} is the complex-to-real spherical
    harmonic transformation matrix. The resulting f_{ℓm}^2 are the real spherical harmonic coefficients
    of |f_s(q)|^2, and are real by construction. Any imaginary part from is
    numerical noise and is discarded before returning.

    The Gaunt array indices (λ, L, ℓ, μ, m) appear in the 3j symbol as (λ L l; μ m-μ -m), and so
    for our purposes, with the physical angular momenta on the RHS of the equalities,
    we have (λ = ℓ1, L = ℓ2, ℓ = ℓ, μ = m1, m = μ). We can also define m2 = m1 - μ.

    # Arguments:
    - R_tensor::Array{Complex{T}, 3}: The R tensor with shape (n_transitions, n_q, n_keys), keyed by key(ℓ, m) = ℓ^2 + (ℓ + m) + 1.
    - gaunt_flm_path::String: Path to the precomputed Gaunt coefficients.

    # Returns:
    - f_lm::Array{T, 3}: The f_{ℓm}^2 tensor with shape (n_transitions, n_q, n_keys),
      keyed identically to the R tensor.
    """

    # Load the Gaunt array.
    gaunt_array = load_gaunt_array(gaunt_flm_path, T)
    gaunt_coeffs = gaunt_array.coefficients

    # Get dimensions.
    n_transitions, n_q, n_keys = size(R_tensor)
    n_gaunt = length(gaunt_coeffs)
    n_threads = nthreads()

    # Allocate the output tensor.
    f_lm = zeros(Complex{T}, n_transitions, n_q, n_keys)

    # Preallocate thread-local accumulation buffers.
    f_lm_local_pool = [zeros(Complex{T}, n_q, n_keys) for _ in 1:n_threads]

    for transition_idx in 1:n_transitions
        # Zero out thread-local buffers for this transition.
        @inbounds for thread_id in 1:n_threads
            fill!(f_lm_local_pool[thread_id], zero(Complex{T}))
        end

        # Extract the R tensor slice for this transition. Use a view to avoid allocations.
        R = @view R_tensor[transition_idx, :, :]

        @threads for gaunt_idx in 1:n_gaunt
            thread_id = threadid()
            f_lm_local = f_lm_local_pool[thread_id]

            # Extract the gaunt entry, with (λ = ℓ1, L = ℓ2, ℓ=ℓ, μ = m1, m = μ).
            l1 = gaunt_array.lambda[gaunt_idx]
            m1 = gaunt_array.mu[gaunt_idx]
            l2 = gaunt_array.L[gaunt_idx]
            l = gaunt_array.l[gaunt_idx]
            mu = gaunt_array.m[gaunt_idx]
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
                    f_lm_local[q_idx, key_out] += combined * R[q_idx, key1] * conj(R[q_idx, key2])
                end
            else
                # U_{μ, μ} and U_{μ, -μ} are the two non-zero entries.
                for m in (mu, -mu)
                    u_val = U(mu, m, T)
                    combined = sign_m2 * gaunt_val * u_val
                    key_out = l * l + (l + m) + 1
                    @inbounds for q_idx in 1:n_q
                        f_lm_local[q_idx, key_out] += combined * R[q_idx, key1] * conj(R[q_idx, key2])
                    end
                end
            end
        end

        # Accumulate thread-local results into the global f_lm tensor.
        @inbounds for f_lm_per_thread in f_lm_local_pool
            f_lm[transition_idx, :, :] .+= f_lm_per_thread
        end
    end

    return real.(f_lm)
end

end
