# This module contains functions to precompute Gaunt coefficients and save the to disk in COO form.

module PrecomputeGaunt

using CGcoefficient, HDF5

include("../utils/FastPowers.jl")
include("../utils/BinEncoding.jl")

using .FastPowers: fast_neg1_pow
using .BinEncoding: encode_bins
using ...SparseTensors
using ...SparseTensors: SparseGauntArray, lm_key

export precompute_gaunt_coefficients

const inv_sqrt_4pi = 1 / sqrt(4π)

function precompute_gaunt_coefficients(l1_max::Int, l2_max::Int, l3_max::Int, output_path::String, ::Type{T} = Float64) where {T<:AbstractFloat}
    """
    Precompute the Gaunt coefficients 𝒢_{ℓ₁m₁,ℓ₂m₂}^{ℓ₃m₃}, defined by

        𝒢_{ℓ₁m₁,ℓ₂m₂}^{ℓ₃m₃} = ∫dΩ Y_ℓ₁^{m₁} Y_ℓ₂^{m₂} conj(Y_ℓ₃^{m₃})
                               = (-1)^{m₃} √((2ℓ₁+1)(2ℓ₂+1)(2ℓ₃+1)/4π)
                                 (ℓ₁ ℓ₂ ℓ₃; m₁ m₂ -m₃) (ℓ₁ ℓ₂ ℓ₃; 0 0 0),

    with m₂ = m₃ - m₁. The two coupled harmonics are the lower indices, and the output
    harmonic is the upper one. The angular momenta obey the selection rules:

        1) |ℓ₁ - ℓ₂| ≤ ℓ₃ ≤ ℓ₁ + ℓ₂,
        2) -ℓ₁ ≤ m₁ ≤ ℓ₁,
        3) -ℓ₃ ≤ m₃ ≤ ℓ₃,
        4) -ℓ₂ ≤ m₂ ≤ ℓ₂,
        5) ℓ₁ + ℓ₂ + ℓ₃ is even.

    The bins group entries by (ℓ₁,m₁), with key(ℓ₁,m₁) = ℓ₁² + ℓ₁ + m₁ + 1.
    Each contraction assigns these indices their physical labels.

    # Arguments:
    - l1_max::Int: The maximum ℓ₁ value to compute Gaunt coefficients for.
    - l2_max::Int: The maximum ℓ₂ value to compute Gaunt coefficients for.
    - l3_max::Int: The maximum ℓ₃ value to compute Gaunt coefficients for.
    - output_path::String: The path to save the computed Gaunt coefficients to (HDF5 format).

    # Returns:
    - Nothing, Gaunt coefficients are saved to disk.
    """

    # Initiialise the 3j symbol calculator. The arguments are max_j, "mode", and what to precompute.
    # Here "3" means just precompute 3j symbols, not 6j etc.
    wigner_init_float(max(l1_max, l2_max, l3_max), "Jmax", 3)

    # Initialise arrays to store the values and COO indices.
    gaunt_coefficients = Float64[]
    l1_indices = Int[]
    l2_indices = Int[]
    l3_indices = Int[]
    m1_indices = Int[]
    m3_indices = Int[]

    # Preallocate l1_m1_bins for all (l1, m1) pairs.
    num_bins = (l1_max + 1) * (l1_max + 1)
    l1_m1_bins = [Int[] for _ in 1:num_bins]

    current_idx = 0

    for l1 in 0:l1_max
        # Precompute the sqrt terms for efficiency.
        sqrt_2l1 = sqrt(2 * l1 + 1)

        for l2 in 0:l2_max
            sqrt_2l2 = sqrt(2 * l2 + 1)

            # Iterate over valid l3 values from selection rule 1.
            for l3 in abs(l1 - l2):min(l1 + l2, l3_max)
                # Skip if the parity rule is violated, selection rule 5.
                if isodd(l1 + l2 + l3)
                    continue
                end

                sqrt_2l3 = sqrt(2 * l3 + 1)
                # Also precompute one of the 3js. Note that f3j takes doubled arguments.
                three_j_zeros = f3j(2 * l1, 2 * l2, 2 * l3, 0, 0, 0)

                # Iterate over valid m1 and m3 values from selection rules 2 and 3.
                for m1 in -l1:l1
                    # Pick out the bin key for this (l1, m1) pair, given by ℓ₁^2 + (ℓ₁ + m₁) + 1.
                    bin_key = lm_key[l1 + 1, m1 + l1 + 1]

                    for m3 in -l3:l3
                        m2 = m3 - m1
                        # Skip if m2 would be invalid from selection rule 4.
                        if m2 < -l2 || m2 > l2
                            continue
                        end

                        # Compute the sign factor (-1)^m3.
                        sign_m3 = fast_neg1_pow(m3, T)

                        # Compute the Gaunt coefficient.
                        Gaunt_value = sign_m3 * sqrt_2l1 * sqrt_2l2 * sqrt_2l3 * inv_sqrt_4pi *
                            three_j_zeros * f3j(2 * l1, 2 * l2, 2 * l3, 2 * m1, 2 * (m3 - m1), -2 * m3)


                        # Store the values and indices.
                        if Gaunt_value != 0.0
                            # Increment the non-zero coefficient index.
                            current_idx += 1
                            push!(gaunt_coefficients, Gaunt_value)
                            push!(l1_indices, l1)
                            push!(l2_indices, l2)
                            push!(l3_indices, l3)
                            push!(m1_indices, m1)
                            push!(m3_indices, m3)

                            # Add this index to the appropriate l1_m1 bin.
                            push!(l1_m1_bins[bin_key], current_idx)
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
        l1_indices,
        m1_indices,
        l2_indices,
        l3_indices,
        m3_indices,
        gaunt_coefficients_typed,
        l1_m1_bins
    )

    # Ensure the output directory exists.
    mkpath(dirname(output_path))

    # Save the Gaunt coefficients to disk in HDF5 format.
    # Encode the bins (vectors of vectors) into flattened arrays for HDF5 storage.
    l1_m1_data, l1_m1_offsets = encode_bins(l1_m1_bins)

    h5open(output_path, "w") do io
        # Write the Gaunt coefficient indices and values.
        write(io, "l1", l1_indices)
        write(io, "m1", m1_indices)
        write(io, "l2", l2_indices)
        write(io, "l3", l3_indices)
        write(io, "m3", m3_indices)
        write(io, "Gaunt_values", gaunt_coefficients_typed)

        # Write the encoded bins as data/offset pairs.
        write(io, "l1_m1_bins/data", l1_m1_data)
        write(io, "l1_m1_bins/offsets", l1_m1_offsets)

        # Also save the maximum angular momenta.
        write(io, "l1_max", l1_max)
        write(io, "l2_max", l2_max)
        write(io, "l3_max", l3_max)
    end
    return println("Gaunt coefficients saved to: ", output_path)
end
end
