# This module provides helpers for splitting a loop range into contiguous per-task chunks.

module ThreadChunks

export chunk_count, chunk_range

@inline function chunk_count(n::Int, n_threads::Int)::Int
    """
    Determine how many chunks to split a range of length n into, so that we never spawn more
    tasks than there is work to do.

    # Arguments:
    - n::Int: The number of items to be processed.
    - n_threads::Int: The number of threads available.

    # Returns:
    - Int: The number of chunks, which is zero when there is no work.
    """
    return n <= 0 ? 0 : min(n_threads, n)
end

@inline function chunk_range(chunk::Int, n_chunks::Int, n::Int)::UnitRange{Int}
    """
    Return the contiguous slice of 1:n belonging to the given chunk, splitting the range as
    evenly as possible. This matches the partitioning that @threads would use, so the memory
    access pattern is unchanged, but it lets each task own its scratch buffers rather than
    indexing a shared pool by threadid(). This helps avoid accidental data races, which can
    occur with the more naive pattern.

    # Arguments:
    - chunk::Int: The 1-based chunk index.
    - n_chunks::Int: The total number of chunks.
    - n::Int: The length of the range being split.

    # Returns:
    - UnitRange{Int}: The subrange of 1:n assigned to this chunk.
    """
    start_idx = div((chunk - 1) * n, n_chunks) + 1
    stop_idx = div(chunk * n, n_chunks)
    return start_idx:stop_idx
end

end