# This module contains functions to precompute Gaunt coefficients and save the to disk in COO form.

module PrecomputeGaunt

using CGcoefficient, HDF5

include("../utils/FastPowers.jl")
include("../utils/BinEncoding.jl")

using .FastPowers: fast_neg1_pow
using .BinEncoding: encode_bins
using ...SparseTensors
using ...SparseTensors: SparseGauntArray, lambda_mu_key

export precompute_gaunt_coefficients

const inv_sqrt_4pi = 1 / sqrt(4π)

function precompute_gaunt_coefficients(lambda_max::Int, L_max::Int, l_max::Int, output_path::String, ::Type{T} = Float64) where {T<:AbstractFloat}
    """
    Precompute Gaunt coefficients defined by:

        G_{λLl}^{μm} = (-1)^m √((2λ + 1)(2L + 1)(2l + 1)/(4π))
                    * (λ L l; 0 0 0) * (λ L l; μ m-μ -m),

    for all valid combinations, and save them to disk in HDF5 format. Here λ_max is set
    by highest angular momentum orbital that we expect in our basis set (λ=0 for s,
    λ=2 for p, etc.). The others are set by our resolution.

    We do not store the values of M as they can be inferred from M = m - μ. The
    coefficients also satisfy the selection rules:

        1) |l - λ| ≤ L ≤ l + λ,
        2) -λ ≤ μ ≤ λ,
        3) -l ≤ m ≤ l.
        4) -L ≤ m - μ ≤ L.
        5) λ + L + l is even.

    The coefficients are stored in a SparseGauntArray structure with lambda_mu_bins
    for efficient lookup during contraction. The bin key for (λ, μ) is computed as:
    key(λ, μ) = λ^2 + (λ + μ) + 1. For λ_max = 6, that's just 49 bins.

    # Arguments:
    - lambda_max::Int: The maximum λ value to compute Gaunt coefficients for.
    - L_max::Int: The maximum L value to compute Gaunt coefficients for.
    - l_max::Int: The maximum l value to compute Gaunt coefficients for.
    - output_path::String: The path to save the computed Gaunt coefficients to (HDF5 format).

    # Returns:
    - Nothing, Gaunt coefficients are saved to disk.
    """

    # Initiialise the 3j symbol calculator. The arguments are max_j, "mode", and what to precompute.
    # Here "3" means just precompute 3j symbols, not 6j etc.
    wigner_init_float(max(lambda_max, L_max, l_max), "Jmax", 3)

    # Initialise arrays to store the values and COO indices.
    gaunt_coefficients = Float64[]
    lambda_indices = Int[]
    L_indices = Int[]
    l_indices = Int[]
    mu_indices = Int[]
    m_indices = Int[]

    # Preallocate lambda_mu_bins for all (lambda, mu) pairs.
    num_bins = (lambda_max + 1) * (lambda_max + 1)
    lambda_mu_bins = [Int[] for _ in 1:num_bins]

    current_idx = 0

    for lambda in 0:lambda_max
        # Precompute the sqrt terms for efficiency.
        sqrt_2lambda = sqrt(2 * lambda + 1)

        for L in 0:L_max
            sqrt_2L = sqrt(2 * L + 1)

            # Iterate over valid l values from selection rule 1.
            for l in abs(lambda - L):min(lambda + L, l_max)
                # Skip if the parity rule is violated, selection rule 5.
                if isodd(lambda + L + l)
                    continue
                end

                sqrt_2l = sqrt(2 * l + 1)
                # Also precompute one of the 3js. Note that f3j takes doubled arguments.
                three_j_zeros = f3j(2 * lambda, 2 * L, 2 * l, 0, 0, 0)

                # Iterate over valid mu and m values from selection rules 2 and 3.
                for mu in -lambda:lambda
                    # Pick out the bin key for this (lambda, mu) pair, given by λ^2 + (λ + μ) + 1.
                    bin_key = lambda_mu_key[lambda + 1, mu + lambda + 1]

                    for m in -l:l
                        M = m - mu
                        # Skip if M would be invalid from selection rule 4.
                        if M < -L || M > L
                            continue
                        end

                        # Compute the sign factor (-1)^m.
                        sign_m = fast_neg1_pow(m, T)

                        # Compute the Gaunt coefficient.
                        G = sign_m * sqrt_2lambda * sqrt_2L * sqrt_2l * inv_sqrt_4pi *
                            three_j_zeros * f3j(2 * lambda, 2 * L, 2 * l, 2 * mu, 2 * (m - mu), -2 * m)


                        # Store the values and indices.
                        if G != 0.0
                            # Increment the non-zero coefficient index.
                            current_idx += 1
                            push!(gaunt_coefficients, G)
                            push!(lambda_indices, lambda)
                            push!(L_indices, L)
                            push!(l_indices, l)
                            push!(mu_indices, mu)
                            push!(m_indices, m)

                            # Add this index to the appropriate lambda_mu bin.
                            push!(lambda_mu_bins[bin_key], current_idx)
                        end
                    end
                end
            end
        end
    end

    # Print the number of non-zero Gaunt coefficients computed.
    println("Number of non-zero Gaunt coefficients: ", length(gaunt_coefficients))

    # Convert the coefficients to the target type.
    gaunt_coefficients_typed = T.(gaunt_coefficients)

    # Create the SparseGauntArray with coefficients of the correct precision.
    gaunt_array = SparseGauntArray(
        lambda_indices,
        mu_indices,
        L_indices,
        l_indices,
        m_indices,
        gaunt_coefficients_typed,
        lambda_mu_bins
    )

    # Ensure the output directory exists.
    mkpath(dirname(output_path))

    # Save the Gaunt coefficients to disk in HDF5 format.
    # Encode the bins (vectors of vectors) into flattened arrays for HDF5 storage.
    lambda_mu_data, lambda_mu_offsets = encode_bins(lambda_mu_bins)

    h5open(output_path, "w") do io
        # Write the Gaunt coefficient indices and values.
        write(io, "lambda", lambda_indices)
        write(io, "mu", mu_indices)
        write(io, "L", L_indices)
        write(io, "l", l_indices)
        write(io, "m", m_indices)
        write(io, "coefficients", gaunt_coefficients_typed)

        # Write the encoded bins as data/offset pairs.
        write(io, "lambda_mu_bins/data", lambda_mu_data)
        write(io, "lambda_mu_bins/offsets", lambda_mu_offsets)

        # Also save the maximum angular momenta.
        write(io, "lambda_max", lambda_max)
        write(io, "L_max", L_max)
        write(io, "l_max", l_max)
    end
    return println("Gaunt coefficients saved to: ", output_path)
end
end
