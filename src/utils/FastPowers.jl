# This module provides functions to efficiently compute powers of -1, i, and general integer powers.

module FastPowers

export fast_i_pow, fast_neg1_pow, pow_int

@inline function pow_int(x::T, n::Int)::T where {T<:AbstractFloat}
    """
    Efficiently compute x^n for small integer powers.

    # Arguments:
    - x::T: The base.
    - n::Int: The exponent.

    # Returns:
    - T: x raised to the power of n.
    """
    if n == 0
        return one(T)
    elseif n == 1
        return x
    elseif n == 2
        return x * x
    elseif n == 3
        return x * x * x
    elseif n == 4
        return (x * x) * (x * x)
    else
        return x ^ n
    end
end

@inline function fast_i_pow(n::Int, ::Type{T}=Float64)::Complex{T} where {T<:AbstractFloat}
    """
    A nice one liner to efficiently compute i^n via:

        i^{1 + 4n} = i, i^{2 + 4n} = -1, i^{3 + 4n} = -i, i^{4 + 4n} = 1,

    for integer n.

    # Arguments:
    - n::Int: The exponent to raise i to.
    - T::Type: The floating point type to use (default: Float64).

    # Returns:
    - i_pow::Complex{T}: The computed i^n value.
    """

    return n % 4 == 0 ? one(T) + zero(T)*im : n % 4 == 1 ? zero(T) + one(T)*im : n % 4 == 2 ? -one(T) + zero(T)*im : zero(T) - one(T)*im
end

@inline function fast_neg1_pow(n::Int, ::Type{T}=Float64)::T where {T<:AbstractFloat}
    """
    A nice one liner to efficiently compute (-1)^n via:

        (-1)^{2n} = 1, (-1)^{2n + 1} = -1,

    for integer n.

    # Arguments:
    - n::Int: The exponent to raise -1 to.
    - T::Type: The floating point type to use (default: Float64).

    # Returns:
    - neg1_pow::T: The computed (-1)^n value.
    """

    return iseven(n) ? one(T) : -one(T)
end

end
