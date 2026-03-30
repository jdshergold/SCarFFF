# This module contains functions to precompute the monomial to spherical tensor, A_{λμ}^{uvw}.

module PrecomputeATensor

using CGcoefficient, HDF5, StaticArrays

include("../utils/FastPowers.jl")
include("../utils/BinEncoding.jl")

using .FastPowers: fast_neg1_pow
using .BinEncoding: encode_bins
using ...SparseTensors
using ...SparseTensors: SparseATensor, uvw_key, lambda_mu_key

export precompute_A_tensor

const SQRT_3o4PI = sqrt(3 / (4π))
const SQRT_4PIo3 = 1 / SQRT_3o4PI
const INV_SQRT_2 = 1 / sqrt(2.0)

# Cartesian to spherical basis coefficients.
# The order is m = {-1, 0, 1}.
const cxm_i = SVector{3, Float64}(INV_SQRT_2, 0.0, -INV_SQRT_2)
const cym_i = SVector{3, ComplexF64}(INV_SQRT_2 * im, 0.0 + 0.0im, INV_SQRT_2 * im)
const czm_i = SVector{3, Float64}(0.0, 1.0, 0.0)

function get_S_uvw(u::Int, v::Int, w::Int, m_i::Tuple{Vararg{Int}})::ComplexF64
    """
    Compute the S^{uvw}({m_i}) coefficient defined as:

        S^{uvw}({m_i}) = (∏_{i=1}^{u1} c_{xm_i}) (∏_{j=u+1}^{u+v} c_{ym_j)} (∏_{k=u+v+1}^{u+v+w} c_{zm_k}),

    where m_i ∈ {-1, 0, 1} correspond to the magnetic quantum numbers of the spherical harmonics
    being added, and the c coefficients are the Cartesian to spherical basis coefficients.

    # Arguments:
    - u::Int: The u index of the S tensor.
    - v::Int: The v index of the S tensor.
    - w::Int: The w index of the S tensor.
    - m_i::Tuple{Vararg{Int}}: Tuple of magnetic quantum numbers. The length should be u + v + w.
       
    # Returns:
    - S_uvw::ComplexF64: The computed S^{uvw}({m_i}) coefficient.
    """

    S_uvw = 1.0 + 0.0im

    # Compute contributions from x components.
    @inbounds for i in 1:u
        # Immediately return zero if there are any m_i = 0.
        m_i[i] == 0 && return 0.0 + 0.0im
        S_uvw *= cxm_i[m_i[i] + 2]
    end

    # Compute contributions from y components.
    @inbounds for j in (u + 1):(u + v)
        # Immediately return zero if there are any m_i = 0.
        m_i[j] == 0 && return 0.0 + 0.0im
        S_uvw *= cym_i[m_i[j] + 2]
    end

    # Compute contributions from z components.
    @inbounds for k in (u + v + 1):(u + v + w)
        # Immediately return zero if there are any m_i != 0.
        m_i[k] != 0 && return 0.0 + 0.0im
        S_uvw *= czm_i[m_i[k] + 2]
    end

    return S_uvw
end

function compute_K_tensor(n_max::Int, m_i::Tuple{Vararg{Int}})::Array{Float64, 2}
    """
    Compute K tensor coefficients arising from spherical harmonic contractions of the form:

        ∏_{i=1}^n Y_1^{m_i} = Σ_{L ∈ {n,n-2,...}} K_L^n({m_i}) Y_L^{M},

    where M = Σ_{i=1}^n m_i. The coefficients can be computed recursively via:

        K_L^1({m_i}) = δ_{L,1},

        K_L^n({m_i}) = Σ_{λ ∈ {n-1,n-3,...}} √(3(2λ+1)(2L+1)/(4π)) (-1)^M_n
                     *  (λ 1 L; M_{n-1} m_n -M_n) (λ 1 L; 0 0 0)
                     *  K_λ^{n-1}({m_i}),

    where M_k = Σ_{i=1}^k m_i. Here the m_i are the magnetic quantum numbers of the 
    harmonics being contracted. These are always zero for L > n.

    # Arguments:
    - n_max::Int: Maximum number of harmonics to contract.
    - m_i::Tuple{Vararg{Int}}: Tuple of magnetic quantum numbers. The length should be n_max.

    # Returns:
    - K_tensor::Array{Float64, 2}: The computed K tensor coefficients.
    """

    # Initialise the 3j symbol calculator. The arguments are max_j, "mode", and what to precompute.
    # Here "3" means just precompute 3j symbols, not 6j etc.
    wigner_init_float(n_max, "Jmax", 3)

    # Initialise K tensor array with zeros. Columns are indexed by λ + 1 to allow λ = 0.
    K_tensor = zeros(Float64, n_max, n_max + 1)

    # Base case for recursion.
    K_tensor[1, 1 + 1] = 1.0
    M_n = m_i[1] # M_1 = m_1.
    M_nm1 = 0 # M_0 = 0.

    # Now compute the K tensor recursively.
    @inbounds for n in 2:n_max
        # Incrmement the magnetic quantum number sums.
        M_n += m_i[n]
        M_nm1 += m_i[n - 1]
        neg1_pow = fast_neg1_pow(M_n, Float64) # (−1)^M_n.

        # L ∈ {n, n-2, ...}.
        L_min = iseven(n) ? 0 : 1
        @inbounds for L in n:-2:L_min
            sqrt_2L = sqrt(2 * L + 1)

            # Accumulate the sum over λ.
            K_n_L = 0.0

            # λ ∈ {n-1, n-3, ...}.
            lambda_min = iseven(n) ? 1 : 0
            @inbounds for lambda in (n - 1):-2:lambda_min
                # Compute the required 3j symbols.
                three_j_M = f3j(2 * lambda, 2, 2 * L, 2 * M_nm1, 2 * m_i[n], -2 * M_n)
                three_j_zeros = f3j(2 * lambda, 2, 2 * L, 0, 0, 0)

                K_n_L += SQRT_3o4PI * sqrt_2L * sqrt(2 * lambda + 1) * neg1_pow * three_j_M * three_j_zeros * K_tensor[n - 1, lambda + 1]
            end
            K_tensor[n, L + 1] = K_n_L
        end
    end

    return K_tensor
end

function decode_mi(mi_idx::Int, n::Int)::Tuple
    """
    Decode mi_idx into a tuple of magnetic quantum numbers, m_i ∈ {-1, 0, 1}.
    This is a clever trick using base 3. Step by step, we have:

        1) The m_i % 3 line picks out the last digit in base 3, {0, 1, 2}.
        2) This is then shifted to {-1, 0, 1} by subtracting 1.
        3) The integer division by 3 shifts to the next bit along.

    As an example, n = 5, mi_idx = 13 in base 3 is "00111", which decodes to m_i = (0, 0, 0, -1, -1).

    # Arguments:
    - mi_idx::Int: Index from 0 to 3^n - 1.
    - n::Int: Number of magnetic quantum numbers.

    # Returns:
    - m_i::Tuple: Tuple of n magnetic quantum numbers.
    """
    m_i = Int[]
    temp_idx = mi_idx
    for _ in 1:n
        m_val = (temp_idx % 3) - 1
        push!(m_i, m_val)
        temp_idx ÷= 3
    end
    return Tuple(m_i)
end

function precompute_A_tensor(n_max::Int, output_path::String, ::Type{T} = Float64) where {T<:AbstractFloat}
    """
    Precompute the A_{λμ}^{uvw} tensor and save it to disk in HDF5 format.
    The A tensor coefficients are defined as:

        A_{λμ}^{uvw} = (4π/3)^(n/2) Σ_{m_i} S^{uvw}({m_i}) K_λ^n({m_i}) δ_{M,μ},

    where the sum is over all combinations of magnetic quantum numbers m_i ∈ {-1, 0, 1}
    for i = 1, ..., n, with n = u + v + w, and M = ∑_{i=1}^n m_i. The non-zero components obey the selection rules:

        1) λ ∈ {n, n-2, n-4, ...},
        2) |μ| ≤ λ.
        3) |μ| ≤ u + v, which arises from the fact that only x and y components increment μ, up to a maximum of u + v.
        4) μ ≡ u + v (mod 2), as μ changes by ±1 for each x or y component.

    We know that for e.g. i-orbitals, n_max = 12, so we can precompute the full A tensor by just speicifying n_max.

    # Arguments:
    - n_max::Int: Maximum degree of our monomials.
    - output_path::String: Path to save the computed A tensor to (HDF5 format).

    # Returns:
    - Nothing, the sparse A tensor is saved to disk.
    """

    # Initialise arrays to store the values and COO indices.
    A_values = ComplexF64[]
    lambda_indices = Int[]
    mu_indices = Int[]
    u_indices = Int[]
    v_indices = Int[]
    w_indices = Int[]

    # Preallocate uvw_bins for all (u, v, w) triplets.
    num_bins = binomial(n_max + 3, 3)
    uvw_bins = [Int[] for _ in 1:num_bins]

    # Preallocate n_bins for all n = u + v + w values.
    # Since u + v + w ≤ n_max, n ranges from 0 to n_max.
    n_bins = [Int[] for _ in 1:(n_max + 1)]

    # Keep track of the current index for COO storage.
    current_idx = 1

    # Loop over all degrees n.
    @inbounds for n in 0:n_max

        # Handle n=0 case separately. This is just q_x^0 q_y^0 q_z^0 = 1.
        if n == 0
            # A_{00}^{000} = √(4π) so that the spherical harmonic Y_0^0 normalization is correct.
            push!(A_values, sqrt(4π) + 0.0im)
            push!(lambda_indices, 0)
            push!(mu_indices, 0)
            push!(u_indices, 0)
            push!(v_indices, 0)
            push!(w_indices, 0)
            uvw_bins[1] = [current_idx]
            n_bins[1] = [current_idx]
            current_idx += 1
            continue
        end

        # Number of m_i combinations for this n.
        num_mi_combinations = 3^n

        # Preallocate array to store K tensors for all m_i combinations.
        # Indices are (n, λ + 1, mi_idx)
        K_n_lambda = zeros(Float64, n, n + 1, num_mi_combinations)

        # Preallocate arrays to cache decoded m_i values and their sums.
        m_i_cache = Vector{Tuple}(undef, num_mi_combinations)
        M_cache = Vector{Int}(undef, num_mi_combinations)

        # Precompute nthe prefactor (4π/3)^(n/2).
        norm_factor = (SQRT_4PIo3)^n

        # Compute K tensors for all m_i combinations for this n.
        @inbounds for mi_idx in 0:(num_mi_combinations - 1)
            m_i = decode_mi(mi_idx, n)
            m_i_cache[mi_idx + 1] = m_i
            M_cache[mi_idx + 1] = sum(m_i)
            K_tensor = compute_K_tensor(n, m_i)
            K_n_lambda[:, :, mi_idx + 1] = K_tensor
        end

        # Preallocate accumulation array for all (λ, μ) pairs, keyed by λ^2 + (λ + μ) + 1.
        num_lambda_mu_bins = (n + 1) * (n + 1)
        lambda_mu_accumulator = zeros(ComplexF64, num_lambda_mu_bins)

        # Now loop over all (u, v, w) triplets with u + v + w = n.
        @inbounds for u in 0:n
            @inbounds for v in 0:(n - u)
                w = n - u - v
                uvw_bin_idx = uvw_key[u + 1, v + 1, w + 1]
                lambda_min = iseven(n) ? 0 : 1

                # Reset the accumulator for this (u, v, w) triplet.
                fill!(lambda_mu_accumulator, 0.0 + 0.0im)

                # Loop over all m_i combinations and accumulate A tensor values.
                @inbounds for mi_idx in 0:(num_mi_combinations - 1)
                    # Look up cached m_i and M values.
                    m_i = m_i_cache[mi_idx + 1]
                    M = M_cache[mi_idx + 1]

                    # Skip if |M| > u + v from selection rule 3.
                    abs(M) > (u + v) && continue

                    # Skip if M and (u + v) have different parity from selection rule 4.
                    isodd(M) != isodd(u + v) && continue

                    # Get S^{uvw}({m_i}) coefficient.
                    S_uvw = get_S_uvw(u, v, w, m_i)

                    # Skip if S_uvw is zero.
                    S_uvw == 0.0 + 0.0im && continue

                    # Loop over valid λ values.
                    @inbounds for lambda in n:-2:lambda_min
                        # Check the selection rule |μ| ≤ λ, with μ = M.
                        abs(M) > lambda && continue

                        # Look up precomputed K value.
                        K_lambda_n = K_n_lambda[n, lambda + 1, mi_idx + 1]

                        # Skip if K is zero.
                        K_lambda_n == 0.0 && continue

                        # Accumulate the A_{λμ}^{uvw} contribution using lambda-mu key.
                        A_lambda_mu_uvw = norm_factor * S_uvw * K_lambda_n
                        lambda_mu_bin_idx = lambda_mu_key[lambda + 1, M + lambda + 1]
                        lambda_mu_accumulator[lambda_mu_bin_idx] += A_lambda_mu_uvw
                    end
                end

                # Store non-zero A tensor values for this (u, v, w) triplet.
                @inbounds for lambda in n:-2:lambda_min
                    @inbounds for mu in -lambda:lambda
                        lambda_mu_bin_idx = lambda_mu_key[lambda + 1, mu + lambda + 1]
                        value = lambda_mu_accumulator[lambda_mu_bin_idx]

                        # Only store non-zero values.
                        if value != 0.0 + 0.0im
                            push!(A_values, value)
                            push!(lambda_indices, lambda)
                            push!(mu_indices, mu)
                            push!(u_indices, u)
                            push!(v_indices, v)
                            push!(w_indices, w)
                            push!(uvw_bins[uvw_bin_idx], current_idx)
                            push!(n_bins[n + 1], current_idx)
                            current_idx += 1
                        end
                    end
                end
            end
        end
    end

    # Print the number of non-zero A tensor coefficients computed.
    println("Number of non-zero A tensor coefficients: ", length(A_values))

    # Convert the coefficients to the target type.
    A_values_typed = Complex{T}.(A_values)

    # Create the sparse tensor with the correct coefficient precision.
    A_tensor = SparseATensor(
        u_indices,
        v_indices,
        w_indices,
        lambda_indices,
        mu_indices,
        A_values_typed,
        uvw_bins,
        n_bins
    )

    # Ensure the output directory exists.
    mkpath(dirname(output_path))

    # Save the A tensor to disk in HDF5 format.
    uvw_data, uvw_offsets = encode_bins(uvw_bins)
    n_data, n_offsets = encode_bins(n_bins)

    h5open(output_path, "w") do io
        # Write the A tensor indices and values.
        write(io, "u", u_indices)
        write(io, "v", v_indices)
        write(io, "w", w_indices)
        write(io, "lambda", lambda_indices)
        write(io, "mu", mu_indices)
        write(io, "A_values", A_values_typed)

        # Write the encoded bins as data/offset pairs.
        write(io, "uvw_bins/data", uvw_data)
        write(io, "uvw_bins/offsets", uvw_offsets)
        write(io, "n_bins/data", n_data)
        write(io, "n_bins/offsets", n_offsets)

        # Also save n_max for reference.
        write(io, "n_max", n_max)
    end

    return println("A tensor saved to: ", output_path)
end
end
