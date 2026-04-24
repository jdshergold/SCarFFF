# This module contains definitions of all sparse, COO structures that we will use in this project.

module SparseTensors

export SparseGauntArray, SparseATensor, SparseDTensor, SparseWTensor, lambda_mu_key, uvw_key

# Precompute the (λ, μ) and (u, v, w) keys, and store them as constant arrays for speed.
# lambda_mu_key goes to λ = 30 (second axis covers μ offset up to 60) so that the f_lm
# Gaunt precomputation, which uses λ up to l_max, works for l_max up to 30.
# uvw_key goes to λ = 12 (i-orbitals), corresponding to i-orbitals.
# These are slightly too large for StaticArrays to be useful.
# Second axis indexes μ via the offset μ + λ, which matches how we look up bins.
const lambda_mu_key = Int[lambda * lambda + mu_offset + 1 for lambda in 0:30, mu_offset in 0:60]
const uvw_key = Int[binomial(u + v + w + 2, 3) + (u * (u + v + w + 1) - (u * (u - 1)) ÷ 2 + v) + 1 for u in 0:12, v in 0:12, w in 0:12]

struct SparseGauntArray{T<:AbstractFloat}
    """
    A structure to store Gaunt coefficients in in sparse COO format.
    - lambda::Vector{Int}: The lambda quantum numbers for each non-zero Gaunt coefficient.
    - mu::Vector{Int}: The mu quantum numbers for each non-zero Gaunt coefficient.
    - L::Vector{Int}: The L quantum numbers for each non-zero Gaunt coefficient.
    - l::Vector{Int}: The l quantum numbers for each non-zero Gaunt coefficient.
    - m::Vector{Int}: The m quantum numbers for each non-zero Gaunt coefficient.
    - coefficients::Vector{T}: The non-zero Gaunt coefficient values.
    - lambda_mu_bins::Vector{Vector{Int}}: Bins of indices for each (lambda, mu) pair to speed up lookups when contracting.

    To clarify the bins. When constructing the Gaunt coefficients we store the index of everything
    entry belonging to a given (λ, μ) in a bin. Then when we want a specific (λ, μ) we can just
    then find the entries corresponding to that via the bin, which itself can be keyed as
    e.g. key(λ,μ) = λ^2 + (λ + μ) + 1. This speeds up lookups significantly. Here the λ^2
    term corresponds to the number of (λ, μ) pairs for all λ' < λ, the (λ + μ) is then the lookup
    for the "current" lambda. The +1 is for 1-based julia indexing.
    """

    lambda::Vector{Int}
    mu::Vector{Int}
    L::Vector{Int}
    l::Vector{Int}
    m::Vector{Int}
    coefficients::Vector{T}
    lambda_mu_bins::Vector{Vector{Int}}
end

struct SparseATensor{T<:AbstractFloat}
    """
    A structure to store A tensor values in sparse COO format.
    - u::Vector{Int}: The u indices for each non-zero A tensor value.
    - v::Vector{Int}: The v indices for each non-zero A tensor value.
    - w::Vector{Int}: The w indices for each non-zero A tensor value.
    - λ::Vector{Int}: The angular momentum quantum numbers for each non-zero A tensor value.
    - μ::Vector{Int}: The magnetic quantum numbers for each non-zero A tensor value.
    - A_values::Vector{Complex{T}}: The non-zero A tensor values.
    - uvw_bins::Vector{Vector{Int}}: Bins of indices for each (u, v, w) triplet to speed up lookups when contracting.
    - n_bins::Vector{Vector{Int}}: Bins of indices for each n = u + v + w value to speed up contractions witn fixed n.

    In the same spirit as the SparseGauntArray, we bin the indices so as to speed up lookups when contracting
    over the uvw indices. Now, for a given n_max, there are far fewer than (n_max + 1)^3 possible (u, v, w)
    triplets, as u + v + w ≤ n_max. In total, there are ∑_n binom(n + 2, 3) = binom(n_max + 3, 3) valid triplets. Then to efficiently
    store the bins, we use the key:

        key(u, v, w) = binom(n + 2, 3) + [u(n+1) + u(u-1)/2 + v] + 1.

    The first term the number of triplets for all n' < n. The square bracketed term comes from as
    follows. At fixed n and u, v can take values {0, ..., n - u}. Then for a given u, the total
    number of (u, v) pairs before that value of u is ∑_{u'=0}^{u-1} (n - u' + 1) = u(n+1) - u(u-1)/2.
    The +v then picks out the value of v wthin that set. For n = 12, this key shrinks the number of bins
    from the naive 2197 to 455. We can also recycle it for the D tensor.

    The n bins provide an additional level of indexing to speed up the contractions with u + v + w = n
    """

    u::Vector{Int}
    v::Vector{Int}
    w::Vector{Int}
    lambda::Vector{Int}
    mu::Vector{Int}
    A_values::Vector{Complex{T}}
    uvw_bins::Vector{Vector{Int}}
    n_bins::Vector{Vector{Int}}
end

struct SparseDTensor{T<:AbstractFloat}
    """
    A structure to store D tensor values in sparse COO format.
    - i::Vector{Int}: The first pair index for each non-zero D tensor value.
    - j::Vector{Int}: The second pair index for each non-zero D tensor value (j >= i, to exploit symmetry).
    - u::Vector{Int}: The u indices for each non-zero D tensor value.
    - v::Vector{Int}: The v indices for each non-zero D tensor value.
    - w::Vector{Int}: The w indices for each non-zero D tensor value.
    - D_values::Vector{Complex{T}}: The non-zero D tensor values.
    - uvw_bins::Vector{Vector{Int}}: Bins of indices for each (u, v, w) triplet to speed up lookups when contracting.
    - n_bins::Vector{Vector{Int}}: Bins of indices for each n = u + v + w value to speed up n-level contractions.
    - ij_bins::Vector{Vector{Int}}: Bins of indices for each (i, j) pair to speed up pair-level contractions.

    Similar to the SparseATensor, we bin the indices so as to speed up lookups when contracting
    over the uvw indices, and for entries with fixed n = u + v + w. We use the same keys as for the SparseATensor.

    We also bin by (i,j) to aid when constructing the W tensor. Since we only store j >= i due to symmetry,
    we use the triangular index key:

        key(i,j) = div(j * (j - 1), 2) + i.

    This will allow us to avoid sorts completely and further reduces memory usage.
    """

    i::Vector{Int}
    j::Vector{Int}
    u::Vector{Int}
    v::Vector{Int}
    w::Vector{Int}
    D_values::Vector{Complex{T}}
    uvw_bins::Vector{Vector{Int}}
    n_bins::Vector{Vector{Int}}
    ij_bins::Vector{Vector{Int}}
end

struct SparseWTensor{T<:AbstractFloat}
    """
    A structure to store W tensor values in sparse COO format.
    - i::Vector{Int}: The first pair index for each non-zero W tensor value.
    - j::Vector{Int}: The second pair index for each non-zero W tensor value (j >= 1, to exploit symmmetry).
    - λ::Vector{Int}: The angular momentum quantum numbers for each non-zero W tensor value.
    - μ::Vector{Int}: The magnetic quantum numbers for each non-zero W tensor value.
    - n::Vector{Int}: The n indices for each non-zero W tensor value.
    - W_values::Vector{Complex{T}}: The non-zero W tensor values.
    - lambda_mu_bins::Vector{Vector{Int}}: Bins of indices for each (λ, μ) pair to speed up lookups when contracting.
    - ij_bins::Vector{Vector{Int}}: Bins of indices for each (i, j) pair to speed up pair-level contractions.
    - n_max::Int: The maximum n value for the W tensor.
    - W_max::T: The global maximum |W_ij| value, used for transition-dependent thresholding.

    The (i,j) and (λ, μ) binning are the same as for the SparseDTensor and SparseGauntArray, respectively.
    """

    i::Vector{Int}
    j::Vector{Int}
    lambda::Vector{Int}
    mu::Vector{Int}
    n::Vector{Int}
    W_values::Vector{Complex{T}}
    lambda_mu_bins::Vector{Vector{Int}}
    ij_bins::Vector{Vector{Int}}
    n_max::Int
    W_max::T
end

end