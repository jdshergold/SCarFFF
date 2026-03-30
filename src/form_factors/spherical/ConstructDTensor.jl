# This module contains functions required to compute the D_{ij}^{uvw} tensor that appears in the spherical decomposition.

module ConstructDTensor

using StaticArrays

using ...SparseTensors

using ..ReadBasisSet: MoleculeData
using ...FastPowers: fast_i_pow
using ...SparseTensors: SparseDTensor, uvw_key

export construct_D_tensor

const factorial_cache = SVector{13, Int}(factorial(n) for n in 0:12)

@inline function get_C_ij_u(
        u::Int,
        b_ij_A::Array{T, 3},
        i::Int,
        j::Int,
        sigma_ij::T,
        max_u::Int,
    )::Complex{T} where {T<:AbstractFloat}
    """
    Compute a single 'rotated' b_{ij}^{u} coefficient, defined as:

        C_{ij}^{u} = i^u ∑_{m=0}^{⌊(a_i + a_j - u)/2⌋} b_{ij}^{u+2m} σ_{ij}^{2(u+m)} (u+2m)!/(m!u!2^m),

    where σ_{ij} is the combined width for Cartesian terms i and j. The v and w indexed C coefficients
    depend instead on the y and z valued b coefficients.

    # Arguments:
    - u::Int: The index to compute the coefficient for.
    - b_ij_A::Array{T,3}: The b_{ij}^{u} coefficients.
    - i::Int: First Cartesian term index.
    - j::Int: Second Cartesian term index.
    - sigma_ij::T: The combined width σ_{ij}.
    - max_u::Int: The maximum u index (a_i + a_j).

    # Returns:
    - C_ij_u::Complex{T}: The computed C_{ij}^{u} coefficient.
    """

    C_ij_u = zero(Complex{T})

    # Compute max m. The function div floors after division.
    max_m = div(max_u - u, 2)

    @inbounds for m in 0:max_m
        C_ij_u +=
            b_ij_A[i, j, u + 2m + 1] * sigma_ij^(2 * (u + m)) * factorial_cache[u + 2m + 1] /
            (factorial_cache[m + 1] * factorial_cache[u + 1] * 2^m)
    end

    return C_ij_u * fast_i_pow(u, T)
end

@inline function compute_and_fill_D_pair!(
        i_indices::Vector{Int},
        j_indices::Vector{Int},
        u_indices::Vector{Int},
        v_indices::Vector{Int},
        w_indices::Vector{Int},
        D_values::Vector{Complex{T}},
        uvw_bins::Vector{Vector{Int}},
        n_bins::Vector{Vector{Int}},
        ij_bins::Vector{Vector{Int}},
        C_u_buffer::Vector{Complex{T}},
        C_v_buffer::Vector{Complex{T}},
        C_w_buffer::Vector{Complex{T}},
        i::Int,
        j::Int,
        max_u::Int,
        max_v::Int,
        max_w::Int,
        M_ij_val::T,
        sigma_ij_val::T,
        b_A::Array{T, 3},
        b_B::Array{T, 3},
        b_C::Array{T, 3},
        current_idx::Ref{Int},
    ) where {T<:AbstractFloat}
    """
    Compute C coefficients and append non-zero D tensor elements for a single ij pair.
    Written as an inplace function to avoid allocations.

    # Arguments:
    - i_indices::Vector{Int}: Vector to store i indices (modified in-place).
    - j_indices::Vector{Int}: Vector to store j indices (modified in-place).
    - u_indices::Vector{Int}: Vector to store u indices (modified in-place).
    - v_indices::Vector{Int}: Vector to store v indices (modified in-place).
    - w_indices::Vector{Int}: Vector to store w indices (modified in-place).
    - D_values::Vector{Complex{T}}: Vector to store D values (modified in-place).
    - uvw_bins::Vector{Vector{Int}}: Bins for (u,v,w) triplets (modified in-place).
    - n_bins::Vector{Vector{Int}}: Bins for n = u+v+w values (modified in-place).
    - C_u_buffer::Vector{Complex{T}}: Pre-allocated buffer for u coefficients.
    - C_v_buffer::Vector{Complex{T}}: Pre-allocated buffer for v coefficients.
    - C_w_buffer::Vector{Complex{T}}: Pre-allocated buffer for w coefficients.
    - i::Int: First Cartesian term index.
    - j::Int: Second Cartesian term index.
    - max_u::Int: Maximum u index (a_i + a_j).
    - max_v::Int: Maximum v index (b_i + b_j).
    - max_w::Int: Maximum w index (c_i + c_j).
    - M_ij_val::T: Cartesian pair weight M_{ij}.
    - sigma_ij_val::T: Combined width σ_{ij}.
    - b_A::Array{T,3}: The b_{ij}^A coefficients.
    - b_B::Array{T,3}: The b_{ij}^B coefficients.
    - b_C::Array{T,3}: The b_{ij}^C coefficients.
    - current_idx::Ref{Int}: Reference to current index counter (modified in-place).
    """

    complex_zero = zero(Complex{T})

    # Compute all C coefficients and store in pre-allocated buffers.
    @inbounds for u in 0:max_u
        C_u_buffer[u + 1] = get_C_ij_u(u, b_A, i, j, sigma_ij_val, max_u)
    end

    @inbounds for v in 0:max_v
        C_v_buffer[v + 1] = get_C_ij_u(v, b_B, i, j, sigma_ij_val, max_v)
    end

    @inbounds for w in 0:max_w
        C_w_buffer[w + 1] = get_C_ij_u(w, b_C, i, j, sigma_ij_val, max_w)
    end

    # Compute D tensor elements and store non-zero values.
    return @inbounds for u in 0:max_u
        C_u_val = C_u_buffer[u + 1]
        @inbounds for v in 0:max_v
            C_v_val = C_v_buffer[v + 1]
            @inbounds for w in 0:max_w
                C_w_val = C_w_buffer[w + 1]
                C_product = C_u_val * C_v_val * C_w_val

                # Only store non-zero values.
                D_ij_value = M_ij_val * C_product
                if D_ij_value != complex_zero
                    push!(i_indices, i)
                    push!(j_indices, j)
                    push!(u_indices, u)
                    push!(v_indices, v)
                    push!(w_indices, w)
                    push!(D_values, D_ij_value)

                    # Add the entry to the appropriate uvw, n, and ij bins.
                    uvw_bin_idx = uvw_key[u + 1, v + 1, w + 1]
                    n_bin_idx = u + v + w + 1
                    ij_bin_idx = div(j * (j - 1), 2) + i
                    push!(uvw_bins[uvw_bin_idx], current_idx[])
                    push!(n_bins[n_bin_idx], current_idx[])
                    push!(ij_bins[ij_bin_idx], current_idx[])
                    current_idx[] += 1
                end
            end
        end
    end
end

function construct_D_tensor(
        mol::MoleculeData{T},
        M_ij::Array{T, 2},
        sigma_ij::Array{T, 2},
        b_A::Array{T, 3},
        b_B::Array{T, 3},
        b_C::Array{T, 3},
    )::SparseDTensor{T} where {T<:AbstractFloat}
    """
    Construct the D_{ij}^{uvw} tensor for all Cartesian pairs of primitives in the molecule.
    This is defined by:

        D_{ij}^{uvw} = M_{ij} * C_{ij}^{u} * C_{ij}^{v} * C_{ij}^{w},

    where u, v, and w range from 0 to a_i + a_j, b_i + b_j, and c_i + c_j, respectively,
    and C_{ij}^{u/v/w} are the 'rotated' b_{ij}^{A/B/C} coefficients.

    # Arguments:
    - mol::MoleculeData: The MoleculeData structure containing molecule and basis set information.
    - M_ij::Array{T,2}: The Cartesian pair weights M_{ij}.
    - sigma_ij::Array{T,2}: The σ_{ij} values.
    - b_A::Array{T,3}: The b_{ij}^A coefficients.
    - b_B::Array{T,3}: The b_{ij}^B coefficients.
    - b_C::Array{T,3}: The b_{ij}^C coefficients.
    # Returns:
    - D_tensor::SparseDTensor: The D_{ij}^{uvw} tensor in sparse COO format.
    """

    n_cartesian_terms = mol.n_cartesian_terms
    max_uvw = size(b_A, 3) - 1 # Max u, v, and w are the same as max A, B and C.

    # Initialise vectors to store sparse tensor data in COO format.
    i_indices = Int[]
    j_indices = Int[]
    u_indices = Int[]
    v_indices = Int[]
    w_indices = Int[]
    D_values = Complex{T}[]

    # Preallocate uvw_bins for all (u, v, w) triplets.
    num_bins = binomial(max_uvw + 3, 3)
    uvw_bins = [Int[] for _ in 1:num_bins]

    # Preallocate n_bins for all n = u + v + w values.
    n_bins = [Int[] for _ in 1:(max_uvw + 1)]

    # Preallocate ij_bins for all (i, j) pairs with j >= i using triangular indexing.
    num_ij_bins = div(n_cartesian_terms * (n_cartesian_terms + 1), 2)
    ij_bins = [Int[] for _ in 1:num_ij_bins]

    # Keep track of the current index for COO storage.
    # This "Ref" object allows us to modify it within functions.
    current_idx = Ref(1)

    # Pre-allocate coefficient buffers once, which are reused for each ij pair.
    C_u_buffer = Vector{Complex{T}}(undef, max_uvw + 1)
    C_v_buffer = Vector{Complex{T}}(undef, max_uvw + 1)
    C_w_buffer = Vector{Complex{T}}(undef, max_uvw + 1)

    @inbounds for i in 1:n_cartesian_terms
        a_i = mol.cartesian_a[i]
        b_i = mol.cartesian_b[i]
        c_i = mol.cartesian_c[i]

        # Since everything is symmetric in i and j, we only need to compute for j >= i.
        @inbounds for j in i:n_cartesian_terms
            a_j = mol.cartesian_a[j]
            b_j = mol.cartesian_b[j]
            c_j = mol.cartesian_c[j]

            max_u = a_i + a_j
            max_v = b_i + b_j
            max_w = c_i + c_j

            sigma_ij_val = sigma_ij[i, j]
            M_ij_val = M_ij[i, j]

            # Compute the C coefficients and fill the D tensor for this ij slice.
            compute_and_fill_D_pair!(
                i_indices,
                j_indices,
                u_indices,
                v_indices,
                w_indices,
                D_values,
                uvw_bins,
                n_bins,
                ij_bins,
                C_u_buffer,
                C_v_buffer,
                C_w_buffer,
                i,
                j,
                max_u,
                max_v,
                max_w,
                M_ij_val,
                sigma_ij_val,
                b_A,
                b_B,
                b_C,
                current_idx,
            )
        end
    end

    # Create the sparse tensor.
    D_tensor = SparseDTensor(
        i_indices,
        j_indices,
        u_indices,
        v_indices,
        w_indices,
        D_values,
        uvw_bins,
        n_bins,
        ij_bins,
    )

    return D_tensor
end
end
