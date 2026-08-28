# This module numerically projects a squared form factor sampled on a (q, θ, ϕ) grid onto real
# spherical harmonic coefficients, giving an f_lm tensor.

module ProjectFLM

using Base.Threads
using SphericalHarmonics
using LinearAlgebra: mul!

using ..ConstructFLMTensor: U
using ...ThreadChunks: chunk_count, chunk_range

export project_f_lm, build_projection_matrices, build_projection_matrix, build_U_blocks,
       project_block!, default_angular_grid

function default_angular_grid(l_max::Int)::Tuple{Int, Int}
    """
    Choose the θ and ϕ grid sizes needed to resolve the form factor at a given l_max.

    The quantity we project is |f|², which is quadratic in the form factor, so its angular content
    runs to 2 * l_max rather than l_max. Nyquist for that is 2 * (2 * l_max) + 1 = 4 * l_max + 1
    samples over a full period.

    # Arguments:
    - l_max::Int: The maximum angular momentum mode of the form factor.

    # Returns:
    - n_theta::Int: The number of θ grid points.
    - n_phi::Int: The number of ϕ grid points.
    """
    l_max >= 0 || error("l_max must not be negative, got $(l_max).")
    points = 4 * l_max + 1
    return points, points
end

function quadrature_weights(grid::Vector{T}, periodic::Bool)::Vector{T} where {T<:AbstractFloat}
    """
    Build composite quadrature weights for a uniform grid.

    For a periodic integrand sampled over a full period with the endpoint repeated, as our ϕ grid is
    over [0, 2π], the trapezoidal rule beats Simpson, so we use it there.
    For θ the integrand is not periodic, so we use composite Simpson, falling back to a trapezoidal
    final interval when the point count is even and Simpson cannot tile the grid exactly.

    # Arguments:
    - grid::Vector{T}: The uniform grid points.
    - periodic::Bool: Whether the integrand is periodic over the full grid.

    # Returns:
    - Vector{T}: The quadrature weights, such that ∫f ≈ Σ_i w_i f(x_i).
    """

    n = length(grid)
    n >= 2 || error("Need at least 2 grid points to integrate, got $(n).")

    h = grid[2] - grid[1]
    weights = zeros(T, n)

    if periodic
        # Trapezoidal on a closed periodic grid: the repeated endpoint gets half weight at each end,
        # which together make one full sample.
        fill!(weights, h)
        weights[1] = h / 2
        weights[n] = h / 2
        return weights
    end

    # Composite Simpson needs an odd number of points; use it over as much of the grid as possible.
    simpson_stop = isodd(n) ? n : n - 1

    if simpson_stop >= 3
        weights[1] += h / 3
        for i in 2:(simpson_stop - 1)
            weights[i] += (iseven(i) ? T(4) : T(2)) * h / 3
        end
        weights[simpson_stop] += h / 3
    end

    # Any leftover final interval (even point count) is handled with a trapezoid.
    if simpson_stop < n
        weights[simpson_stop] += h / 2
        weights[n] += h / 2
    end

    return weights
end

# Key for (ℓ, m ≥ 0) coefficients.
@inline positive_key(l::Int, m::Int) = (l * (l + 1)) ÷ 2 + m + 1


function build_projection_matrix(
        theta_grid::Vector{T},
        phi_grid::Vector{T},
        l_max::Int,
    )::Matrix{T} where {T<:AbstractFloat}
    """
    Build the projection matrix that takes |f(vec q)|² to its complex harmonic
    coefficients, keeping only μ ≥ 0 and stacking the real and imaginary parts into one matrix.
    The μ < 0 components can be found from c_{ℓ,-μ} = (-1)^μ conj(c_{ℓμ}).

    # Arguments:
    - theta_grid::Vector{T}: The θ grid, in radians.
    - phi_grid::Vector{T}: The ϕ grid, in radians.
    - l_max::Int: The maximum angular momentum mode.

    # Returns:
    - Matrix{T}: The stacked matrix, dimensions (2 * n_keys_pos, n_theta * n_phi).
    """

    # Build the projection matrices, and the key for the positive m part of the matrix.
    A_real, A_imag = build_projection_matrices(theta_grid, phi_grid, l_max)
    n_keys_pos = ((l_max + 1) * (l_max + 2)) ÷ 2
    n_points = size(A_real, 2)

    A = Matrix{T}(undef, 2 * n_keys_pos, n_points)
    @inbounds for l in 0:l_max, m in 0:l
        source = l * l + l + m + 1
        target = positive_key(l, m)
        for point_idx in 1:n_points
            # Put the real and imaginary parts into a single matrix, n_keys apart.
            A[target, point_idx] = A_real[source, point_idx]
            A[n_keys_pos + target, point_idx] = A_imag[source, point_idx]
        end
    end

    return A
end


function build_projection_matrices(
        theta_grid::Vector{T},
        phi_grid::Vector{T},
        l_max::Int,
    )::Tuple{Matrix{T}, Matrix{T}} where {T<:AbstractFloat}
    """
    Build the weighted conjugate spherical harmonic matrices used to project onto complex harmonic
    coefficients,

        c_{ℓμ}(q) = ∫ dΩ |f(q, Ω)|² conj(Y_ℓ^μ(Ω)),

    discretised as c = A |f|² with A[key, p] = w_θ sinθ w_ϕ conj(Y_ℓ^μ(θ_p, ϕ_p)).

    The real and imaginary parts are returned separately so the projection can be done with two real
    matrix multiplications: |f|² is real, so real BLAS is both faster and avoids allocating a complex
    copy of the (potentially large) form factor array.

    # Arguments:
    - theta_grid::Vector{T}: The θ grid, in radians.
    - phi_grid::Vector{T}: The ϕ grid, in radians.
    - l_max::Int: The maximum angular momentum mode.

    # Returns:
    - A_real::Matrix{T}: The real part, with dimensions (n_keys, n_theta * n_phi).
    - A_imag::Matrix{T}: The imaginary part, with the same dimensions.
    """

    # Get the grid sizes and number of keys.
    n_theta = length(theta_grid)
    n_phi = length(phi_grid)
    n_keys = (l_max + 1)^2
    n_points = n_theta * n_phi

    # Get the integration weights.
    theta_weights = quadrature_weights(theta_grid, false)
    phi_weights = quadrature_weights(phi_grid, true)

    # Allocate matrices to hold the real and imaginary parts of the non-f part of the integral.
    A_real = Matrix{T}(undef, n_keys, n_points)
    A_imag = Matrix{T}(undef, n_keys, n_points)

    n_chunks = chunk_count(n_theta, nthreads())

    @sync for chunk in 1:n_chunks
        Threads.@spawn begin
            # Each task owns its harmonic cache.
            Ylm_cache = SphericalHarmonics.cache(l_max, SphericalHarmonics.FullRange)

            for theta_idx in chunk_range(chunk, n_chunks, n_theta)
                theta = theta_grid[theta_idx]
                # The sinθ Jacobian of the solid angle element forms part of the theta weight.
                theta_weight = theta_weights[theta_idx] * sin(theta)

                computePlmcostheta!(Ylm_cache, theta, l_max)

                for phi_idx in 1:n_phi
                    phi = phi_grid[phi_idx]
                    # Compute the combined weight for this (θ, ϕ) point, which is the product of the θ and ϕ weights.
                    weight = theta_weight * phi_weights[phi_idx]

                    computeYlm!(Ylm_cache, theta, phi, l_max)
                    Yvals = SphericalHarmonics.getY(Ylm_cache)

                    point_idx = (phi_idx - 1) * n_theta + theta_idx
                    @inbounds for l in 0:l_max
                        key_base = l * l + l + 1
                        for m in -l:l
                            # Add in the spherical harmonic part to form A.
                            value = weight * conj(Complex{T}(Yvals[(l, m)]))
                            A_real[key_base + m, point_idx] = real(value)
                            A_imag[key_base + m, point_idx] = imag(value)
                        end
                    end
                end
            end
        end
    end

    return A_real, A_imag
end

function build_U_blocks(l_max::Int, ::Type{T}) where {T<:AbstractFloat}
    """
    Precompute the complex-to-real spherical harmonic transformation, one block per l.

    # Arguments:
    - l_max::Int: The maximum angular momentum mode.
    - T::Type: The floating point type to use.

    # Returns:
    - Vector{Matrix{Complex{T}}}: The U block for each l, indexed [μ + l + 1, m + l + 1].
    """
    return [Matrix{Complex{T}}([U(mu, m, T) for mu in -l:l, m in -l:l]) for l in 0:l_max]
end

function project_block!(
        f_lm::Array{T, 3},
        f_sq_block::AbstractArray{T, 4},
        A::Matrix{T},
        U_blocks::Vector{Matrix{Complex{T}}},
        q_offset::Int,
        l_max::Int;
        occupancy::Union{Nothing, Matrix{Bool}} = nothing,
    ) where {T<:AbstractFloat}
    """
    Project one block of q points and accumulate the result into the output f_lm tensor.

    Splitting the projection this way streams over q rather than materialising the
    whole squared form factor, which could be very large.

    The first axis is either the transition index for the form factor, or the energy value for the structure function.

    # Arguments:
    - f_lm::Array{T, 3}: The output tensor, with dimensions (n_states, n_q, n_keys).
    - f_sq_block::AbstractArray{T, 4}: The squared form factor for this block, with dimensions (n_states, n_q_block, n_theta, n_phi).
    - A::Matrix{T}: The stacked (real, then imaginary, n_keys apart) μ ≥ 0 projection matrix from build_projection_matrix.
    - U_blocks::Vector{Matrix{Complex{T}}}: The complex-to-real transformation blocks.
    - q_offset::Int: The index in f_lm of the first q point in this block.
    - l_max::Int: The maximum angular momentum mode.
    - occupancy::Union{Nothing, Matrix{Bool}}: Which (state, q) of this block hold anything, with
      dimensions (n_states, n_q_block). The rest are skipped, as they are exactly zero.

    # Returns:
    - Nothing. f_lm is modified in place.
    """

    # Get the block sizes.
    n_states, n_q_block, n_theta, n_phi = size(f_sq_block)
    n_points = n_theta * n_phi

    # Take the columns that hold something, keeping the (q, state) order the unmasked path uses.
    column_q = Vector{Int}(undef, n_states * n_q_block)
    column_state = Vector{Int}(undef, n_states * n_q_block)
    n_columns = 0
    # For the structure function, state is now the energy value.
    @inbounds for q_local in 1:n_q_block, state_idx in 1:n_states
        (occupancy === nothing || occupancy[state_idx, q_local]) || continue # Skip empty energy values.
        n_columns += 1
        column_q[n_columns] = q_local
        column_state[n_columns] = state_idx
    end
    n_columns == 0 && return nothing

    # A Julia trick, stops lots of lookups later.
    used_columns = n_columns

    # Flatten the kept (state, q) into columns and (θ, ϕ) into rows, matching the point ordering used
    # when the projection matrices were built.
    B = Matrix{T}(undef, n_points, n_columns)
    @inbounds for phi_idx in 1:n_phi, theta_idx in 1:n_theta
        point_idx = (phi_idx - 1) * n_theta + theta_idx
        for column in 1:n_columns
            B[point_idx, column] =
                f_sq_block[column_state[column], column_q[column], theta_idx, phi_idx]
        end
    end

    # One real GEMM gives the real and imaginary parts of the μ ≥ 0 coefficients together. |f|² is
    # real, so keeping BLAS in real arithmetic is both faster and avoids a complex copy of the input.
    n_keys_pos = ((l_max + 1) * (l_max + 2)) ÷ 2
    C = Matrix{T}(undef, 2 * n_keys_pos, n_columns)
    mul!(C, A, B)

    # Rotate the complex coefficients into the real harmonic basis. We thread over columns,
    # which are either (q, E) for the structure function, or (q, transition) for the form factor.
    n_chunks = chunk_count(used_columns, nthreads())
    @sync for chunk in 1:n_chunks
        Threads.@spawn begin
            # μ < 0 is found once per (column, ℓ) from c_{ℓ,-μ} = (-1)^μ conj(c_{ℓμ}).
            coefficients = Vector{Complex{T}}(undef, 2 * l_max + 1)

            for column in chunk_range(chunk, n_chunks, used_columns)
                q_idx = q_offset + column_q[column] - 1
                state_idx = column_state[column]

                @inbounds for l in 0:l_max
                    key_base = l * l + l + 1
                    U_l = U_blocks[l + 1]

                    for mu in 0:l
                        key = positive_key(l, mu)
                        coefficient = Complex{T}(C[key, column], C[n_keys_pos + key, column])
                        coefficients[mu + l + 1] = coefficient
                        mu == 0 && continue
                        coefficients[l + 1 - mu] =
                            isodd(mu) ? -conj(coefficient) : conj(coefficient)
                    end

                    for m in -l:l
                        total = zero(Complex{T})
                        for mu in 1:(2 * l + 1)
                            total += coefficients[mu] * U_l[mu, m + l + 1]
                        end
                        # |f|² is real, so f_lm is real by construction and any imaginary part is noise.
                        f_lm[state_idx, q_idx, key_base + m] = real(total)
                    end
                end
            end
        end
    end

    return nothing
end

function project_f_lm(
        f_sq::Array{T, 4},
        theta_grid::Vector{T},
        phi_grid::Vector{T},
        l_max::Int;
        q_block::Int = 32,
    )::Array{T, 3} where {T<:AbstractFloat}
    """
    Project |f(q, θ, ϕ)|² onto real spherical harmonic coefficients, producing an f_lm tensor,
    keyed by key(ℓ, m) = ℓ² + (ℓ + m) + 1.

    We first project onto complex harmonic coefficients by quadrature, then rotate into the real
    basis with the same U matrix the analytic Gaunt path uses, so the convention matches:

        c_{ℓμ}(q) = ∫ dΩ |f|² conj(Y_ℓ^μ),        f²_{ℓm}(q) = Σ_μ c_{ℓμ}(q) U_{μm}.

    The quadrature is expressed as a matrix product over the flattened angular grid so that it runs
    through BLAS rather than a loop, which is the difference between seconds and
    minutes for large grid sizes.

    # Arguments:
    - f_sq::Array{T, 4}: The squared form factor, with dimensions (n_states, n_q, n_theta, n_phi).
    - theta_grid::Vector{T}: The θ grid, in radians.
    - phi_grid::Vector{T}: The ϕ grid, in radians.
    - l_max::Int: The maximum angular momentum mode.
    - q_block::Int: How many q points to project at once (default: 32).

    # Returns:
    - Array{T, 3}: The f_lm tensor, with dimensions (n_states, n_q, n_keys).
    """

    # Get the block sizes and number of keys.
    n_states, n_q, n_theta, n_phi = size(f_sq)
    length(theta_grid) == n_theta || error("θ grid has $(length(theta_grid)) points but the form factor has $(n_theta).")
    length(phi_grid) == n_phi || error("ϕ grid has $(length(phi_grid)) points but the form factor has $(n_phi).")

    n_keys = (l_max + 1)^2

    # Build the projection matrices and U blocks.
    # The U blocks map from complex to real spherical harmonics.
    A = build_projection_matrix(theta_grid, phi_grid, l_max)
    U_blocks = build_U_blocks(l_max, T)

    f_lm = Array{T, 3}(undef, n_states, n_q, n_keys)

    # Project one block of q points at a time.
    for q_start in 1:q_block:n_q
        q_stop = min(q_start + q_block - 1, n_q)
        block = f_sq[:, q_start:q_stop, :, :]
        project_block!(f_lm, block, A, U_blocks, q_start, l_max)
    end

    return f_lm
end

end
