# This module provides functions to compute probabilist's Hermite polynomials He_n(x) from n = 0 up to 6.

module ProbabilistsHermite

export fill_hermite_vector!

@inline function fill_hermite_vector!(He::Vector{T}, max_order::Int, x::T) where {T<:Number}
    """
    Fill a pre-allocated buffer with probabilist's Hermite polynomials from He_0(x) to He_{max_order}(x)
    using the recurrence relation:

        He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x).

    The probabilist's Hermite polynomials are related to the physicist's Hermite polynomials
    H_n(x), by:

        He_n(x) = 2^(-n/2) * H_n(x/√2).

    # Arguments:
    - He::Vector{T}: Pre-allocated buffer to store the results.
    - max_order::Int: The maximum polynomial order to compute, which must be less than or equal to 6.
    - x::T: The point at which to evaluate the polynomials.

    # Returns:
    - Nothing. Results are written to the He buffer in-place.
    """

    # Start with the base case, He_0(x) = 1.
    He[1] = one(T)

    if max_order >= 1
        He[2] = x  # He_1(x) = x.
    end

    # Now use the recurrence relation for the remaining polynomials.
    for n in 1:min(max_order-1, 5)
        He[n+2] = x * He[n+1] - n * He[n]
    end

    return nothing
end

end