module BinEncoding

export encode_bins, decode_bins

@inline function encode_bins(bins::Vector{Vector{Int}})
    """
    Flatten a vector of bins (vector of vectors) into a single data array and an offset array for HDF5 storage.

    HDF5 does not natively support vectors of vectors, so we encode them as:
    - data: A flattened 1D array containing all bin elements.
    - offsets: An array of indices marking where each bin starts in the data array, as well as the end of the last bin.

    For example, bins = [[1, 5, 7], [2, 3], [4, 6, 8, 9]] becomes:
    - data = [1, 5, 7, 2, 3, 4, 6, 8, 9]
    - offsets = [1, 4, 6, 10]

    # Arguments:
    - bins::Vector{Vector{Int}}: A vector of bins to encode.

    # Returns:
    - data::Vector{Int}: The flattened data array.
    - offsets::Vector{Int}: The offset array.
    """
    # Preallocate the offsets array.
    offsets = Vector{Int}(undef, length(bins) + 1)
    offsets[1] = 1
    total = 0
    # Fill the offsets array.
    @inbounds for (bin_idx, bin) in enumerate(bins)
        total += length(bin)
        offsets[bin_idx + 1] = total + 1
    end
    # Preallocate the data array.
    data = Vector{Int}(undef, total)
    pos = 1
    @inbounds for bin in bins
        len_bin = length(bin)
        if len_bin > 0
            data[pos:(pos + len_bin - 1)] .= bin
        end
        pos += len_bin
    end
    return data, offsets
end

@inline function decode_bins(data::Vector{Int}, offsets::Vector{Int})
    """
    Reconstruct a vector of bins from flattened HDF5 data and offset arrays.

    This is the inverse of encode_bins. Given a flattened data array and offsets,
    reconstruct the original vector of vectors.

    For example, data = [1, 5, 7, 2, 3, 4, 6, 8, 9] and offsets = [1, 4, 6, 10] yields:
    bins = [[1, 5, 7], [2, 3], [4, 6, 8, 9]].

    # Arguments:
    - data::Vector{Int}: The flattened data array.
    - offsets::Vector{Int}: The offset array.

    # Returns:
    - bins::Vector{Vector{Int}}: The reconstructed vector of bins.
    """
    n_bins = length(offsets) - 1
    # Allocate the output bins.
    bins = Vector{Vector{Int}}(undef, n_bins)
    @inbounds for i in 1:n_bins
        # Get the first and last indices for this bin.
        start_idx = offsets[i]
        end_idx = offsets[i + 1] - 1
        # Extract the bin data.
        if start_idx <= end_idx
            bins[i] = data[start_idx:end_idx]
        else
            bins[i] = Int[]
        end
    end
    return bins
end

end