# This module contains functions to compute the W tensor in sparse COO format.

module ConstructWTensor

using HDF5
using Base.Threads

include("../../utils/BinEncoding.jl")

using .BinEncoding: decode_bins
using ...SparseTensors
using ...SparseTensors: SparseDTensor, SparseWTensor, SparseATensor, lambda_mu_key, uvw_key

@inline function load_sparse_A_tensor(path::String, ::Type{T}) where {T<:AbstractFloat}
    """
    Load a SparseATensor from an HDF5 file.

    This function reads the A tensor coefficients and associated indices from a HDF5 file
    and reconstructs the SparseATensor structure. The bins are decoded from the
    flattened HDF5 representation.

    # Arguments:
    - path::String: Path to the HDF5 file containing the A tensor.
    - T::Type: The floating point type to use for the coefficients.

    # Returns:
    - SparseATensor{T}: The loaded A tensor.
    """
    h5open(path, "r") do io
        # Read in the indices and coefficients.
        u = Vector{Int}(read(io, "u"))
        v = Vector{Int}(read(io, "v"))
        w = Vector{Int}(read(io, "w"))
        lambda = Vector{Int}(read(io, "lambda"))
        mu = Vector{Int}(read(io, "mu"))
        A_values = Vector{Complex{T}}(read(io, "A_values"))
        # Reconstruct the uvw and n bins.
        uvw_bins = decode_bins(Vector{Int}(read(io, "uvw_bins/data")), Vector{Int}(read(io, "uvw_bins/offsets")))
        n_bins = decode_bins(Vector{Int}(read(io, "n_bins/data")), Vector{Int}(read(io, "n_bins/offsets")))
        return SparseATensor(u, v, w, lambda, mu, A_values, uvw_bins, n_bins)
    end
end

function construct_W_tensor(D_tensor::SparseDTensor{T}, A_tensor_path::String; threshold::T = 0.0)::SparseWTensor{T} where {T<:AbstractFloat}
    """
    Construct the W tensor defined by:

        W_{ij,λμ}^{n} = ∑_{u,v,w,u+v+w=n} D_{ij}^{uvw} A_{uvw}^{λμ},

    where D_{ij}^{uvw} is the D tensor and A_{uvw}^{λμ} is the A tensor, both stored in sparse COO format.
    The resulting W tensor is also stored in sparse COO format.

    # Arguments:
    - D_tensor::SparseDTensor: The D tensor in sparse COO format.
    - A_tensor_path::String: The path to the precomputed A tensor, which is stored in HDF5 format.
    - threshold::T: A threshold value below which W tensor entries satisfying (|W/W_max| < threshold) will be discarded (default: 0.0, so no discarding).

    # Returns:
    - W_tensor::SparseWTensor: The constructed W tensor in sparse COO format.
    """

    typed_complex_zero = zero(Complex{T})

    # Load the precomputed A tensor from disk.
    A_tensor = load_sparse_A_tensor(A_tensor_path, T)
    A_values = A_tensor.A_values

    # Get the tensor dimensions.
    n_max = length(A_tensor.n_bins) - 1
    n_dim = n_max + 1
    lambda_dim = n_max + 1
    mu_dim = 2 * n_max + 1

    # Compute the mu offset for indexing. This way e.g. μ = -n_max maps to index 1, and μ = n_max maps to index 2 * n_max + 1.
    mu_offset = n_max + 1

    num_ij_bins = length(D_tensor.ij_bins)
    num_A_uvw_bins = length(A_tensor.uvw_bins)

    # Get the number of threads and allocate thread-local storage.
    n_threads = nthreads()

    # Allocate thread-local buffers to store W tensor entries in COO format.
    i_indices_pool = [Int[] for _ in 1:n_threads]
    j_indices_pool = [Int[] for _ in 1:n_threads]
    lambda_indices_pool = [Int[] for _ in 1:n_threads]
    mu_indices_pool = [Int[] for _ in 1:n_threads]
    n_indices_pool = [Int[] for _ in 1:n_threads]
    W_values_pool = [Complex{T}[] for _ in 1:n_threads]

    # Allocate thread-local accumulators for each (i, j) slice.
    W_ij_slice_pool = [zeros(Complex{T}, n_dim, lambda_dim, mu_dim) for _ in 1:n_threads]
    nonzero_slots_pool = [Vector{NTuple{3, Int}}(undef, n_dim * lambda_dim * mu_dim) for _ in 1:n_threads]

    # Process (i, j) bins in parallel.
    @threads for bin_idx in 1:num_ij_bins
        ij_bin = D_tensor.ij_bins[bin_idx]

        # If the bin is empty, we can skip it.
        isempty(ij_bin) && continue

        # Get the thread ID.
        thread_id = threadid()

        # Track the number of nonzero slots for this (i, j) bin.
        nonzero_count = 0

        # Get the corresponding slice indices for the bin.
        first_idx = ij_bin[1]
        pair_i = D_tensor.i[first_idx]
        pair_j = D_tensor.j[first_idx]

        # Now loop over all D tensor entries in this (i, j) bin.
        @inbounds for D_idx in ij_bin

            # Get the (u, v, w) indices for this D tensor entry.
            # We use these to contract with the correct A tensor entries.
            u = D_tensor.u[D_idx]
            v = D_tensor.v[D_idx]
            w = D_tensor.w[D_idx]
            uvw_idx = uvw_key[u + 1, v + 1, w + 1]

            uvw_idx > num_A_uvw_bins && continue

            A_bin = A_tensor.uvw_bins[uvw_idx]

            # If there are no entries for this bin, skip this D bin.
            isempty(A_bin) && continue

            D_val = D_tensor.D_values[D_idx]
            n = u + v + w
            n > n_max && continue

            # Now loop over the matching A tensor entries and accumulate into the W_ij_slice.
            @inbounds for A_idx in A_bin
                lambda = A_tensor.lambda[A_idx]
                mu = A_tensor.mu[A_idx]
                A_val = A_values[A_idx]

                # Skip if λ > n.
                lambda > n && continue

                # Accumulate into W_ij_slice.
                n_idx = n + 1
                lambda_idx = lambda + 1
                mu_idx = mu + mu_offset

                # Keep track of which slots have nonzero entries.
                if W_ij_slice_pool[thread_id][n_idx, lambda_idx, mu_idx] == typed_complex_zero
                    nonzero_count += 1
                    nonzero_slots_pool[thread_id][nonzero_count] = (n_idx, lambda_idx, mu_idx)
                end
                W_ij_slice_pool[thread_id][n_idx, lambda_idx, mu_idx] += D_val * A_val
            end
        end

        # Skip the pushing to W_ij if there are no nonzero entries.
        nonzero_count == 0 && continue

        # Push the nonzero entries to the thread-local W tensor and reset the slice and counters.
        @inbounds for idx in 1:nonzero_count
            n_idx, lambda_idx, mu_idx = nonzero_slots_pool[thread_id][idx]
            value = W_ij_slice_pool[thread_id][n_idx, lambda_idx, mu_idx]
            value == typed_complex_zero && continue
            push!(i_indices_pool[thread_id], pair_i)
            push!(j_indices_pool[thread_id], pair_j)
            push!(lambda_indices_pool[thread_id], lambda_idx - 1)
            push!(mu_indices_pool[thread_id], mu_idx - mu_offset)
            push!(n_indices_pool[thread_id], n_idx - 1)
            push!(W_values_pool[thread_id], value)
            W_ij_slice_pool[thread_id][n_idx, lambda_idx, mu_idx] = typed_complex_zero
        end
    end

    # Merge the thread-local results into global arrays.
    i_indices = reduce(vcat, i_indices_pool)
    j_indices = reduce(vcat, j_indices_pool)
    lambda_indices = reduce(vcat, lambda_indices_pool)
    mu_indices = reduce(vcat, mu_indices_pool)
    n_indices = reduce(vcat, n_indices_pool)
    W_values = reduce(vcat, W_values_pool)

    # Discard anything less than the threshold if specified.
    if threshold != 0
        max_W_value = maximum(abs.(W_values))
        threshold = max_W_value * T(threshold)
        filtered_indices = findall(abs.(W_values) .>= threshold)
        i_indices = i_indices[filtered_indices]
        j_indices = j_indices[filtered_indices]
        lambda_indices = lambda_indices[filtered_indices]
        mu_indices = mu_indices[filtered_indices]
        n_indices = n_indices[filtered_indices]
        W_values = W_values[filtered_indices]
    end

    # Build (λ, μ) bins.
    max_lambda = maximum(lambda_indices)
    num_lambda_mu_bins = (max_lambda + 1) * (max_lambda + 1)
    lambda_mu_bins = [Int[] for _ in 1:num_lambda_mu_bins]

    @inbounds for idx in 1:length(i_indices)
        lambda = lambda_indices[idx]
        mu = mu_indices[idx]
        bin_key = lambda_mu_key[lambda + 1, mu + lambda + 1]
        push!(lambda_mu_bins[bin_key], idx)
    end

    # Also build the (i, j) bins using triangular indexing (j >= i).
    ij_bins = [Int[] for _ in 1:num_ij_bins]

    @inbounds for idx in 1:length(i_indices)
        i = i_indices[idx]
        j = j_indices[idx]
        bin_key = div(j * (j - 1), 2) + i
        push!(ij_bins[bin_key], idx)
    end

    # Compute and store the global maximum |W_ij| for future thresholding.
    W_max = maximum(abs.(W_values))

    return SparseWTensor(i_indices, j_indices, lambda_indices, mu_indices, n_indices, W_values, lambda_mu_bins, ij_bins, n_max, W_max)
end

end
