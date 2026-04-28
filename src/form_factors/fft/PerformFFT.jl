module PerformFFT

using FFTW
using Base.Threads
using Integrals

export perform_fft, check_parseval_theorem

function perform_fft(
        transition_densities::Array{T, 4},
        r_lim::Vector{U},
        N_grid::Vector{Int}
    )::Array{Complex{T}, 4} where {T<:AbstractFloat, U<:AbstractFloat}
    """
    Perform a 3D FFT on the transition density data to get the form factor.

    # Arguments:
    - transition_densities::Array{T, 4}: The 4D transition density data to transform with shape (n_transitions, n_x, n_y, n_z).
    - r_lim::Vector{U}: The limits of the grid in real space, in Angstroms.
    - N_grid::Vector{Int}: The number of grid points in each dimension.

    # Returns:
    - Array{Complex{T}, 4}: The Fourier-transformed transition densities (form factors in momentum space) with shape (n_transitions, n_qx, n_qy, n_qz).
    """

    # Extract the number of transitions.
    n_transitions = size(transition_densities, 1)

    # First construct the spatial and momentum grids for the phase factor.
    r_res = 2 .* r_lim ./ N_grid

    xs = T.(collect(range(-r_lim[1], step = r_res[1], length=N_grid[1])))
    ys = T.(collect(range(-r_lim[2], step = r_res[2], length=N_grid[2])))
    zs = T.(collect(range(-r_lim[3], step = r_res[3], length=N_grid[3])))

    # We don't shift these, and keep them in the same units for the phase factor.
    qxs = T(2π) .* T.(collect(fftfreq(N_grid[1], 1/r_res[1])))
    qys = T(2π) .* T.(collect(fftfreq(N_grid[2], 1/r_res[2])))
    qzs = T(2π) .* T.(collect(fftfreq(N_grid[3], 1/r_res[3])))

    # Setup multithreading in FFTW.
    FFTW.set_num_threads(nthreads())

    # Compute the integral of the real space form factor.
    dV = prod(r_res)

    # Allocate the output array.
    form_factors = Array{Complex{T}, 4}(undef, n_transitions, N_grid[1], N_grid[2], N_grid[3])

    # Process each transition sequentially.
    for t in 1:n_transitions
        # Extract this transition's density and promote it to complex.
        transition_density = Complex{T}.(view(transition_densities, t, :, :, :))

        # Perform the FFT in place for memory.
        fft!(transition_density)

        # Conjugate to partially correct phase.
        conj!(transition_density)

        # Now multiply by the phase factor exp(iq.r_min) for each grid point. Multithread for speed.
        x0, y0, z0 = xs[1], ys[1], zs[1] # For computing the phase factor.
        @threads for k in 1:N_grid[3]
            z_prod = qzs[k] * z0
            @inbounds for j in 1:N_grid[2]
                y_prod = qys[j] * y0
                for i in 1:N_grid[1]
                    transition_density[i,j,k] *= exp(im * (qxs[i] * x0 + y_prod + z_prod))
                end
            end
        end

        # Multiply by voxel volume. The form factor should now be normalised appropriately.
        transition_density .*= T(dV)

        # Shift the zero-frequency component to the center.
        transition_density .= fftshift(transition_density)

        # Store the result.
        form_factors[t, :, :, :] .= transition_density
    end

    return form_factors
end

function check_parseval_theorem(
        transition_density::Array{T, 3},
        form_factor::Array{Complex{T}, 3},
        r_lim::Vector{U},
        N_grid::Vector{Int}
    )::Nothing where {T<:AbstractFloat, U<:AbstractFloat}
    """
    Check that Parseval's theorem holds for the FFT. That is, we check that:

        ∫ |ρ(r)|^2 d^3r = (1/(2π)³) ∫ |f_s(q)|^2 d^3q,

    holds where ρ(r) is the transition density and f_s(q) is the form factor.

    # Arguments:
    - transition_density::Array{T, 3}: The transition density.
    - form_factor::Array{Complex{T}, 3}: The form factor.
    - r_lim::Vector{U}: The limits of the grid in real space, in Angstroms.
    - N_grid::Vector{Int}: The number of grid points in each dimension.

    # Returns:
    - Nothing. Prints the comparison of real and momentum space integrals.
    """

    # Compute the spatial and momentum resolutions.
    r_res = 2 .* r_lim ./ N_grid

    # Compute the q-grid.
    qxs = T(2π) .* T.(collect(fftfreq(N_grid[1], 1/r_res[1])))
    qys = T(2π) .* T.(collect(fftfreq(N_grid[2], 1/r_res[2])))
    qzs = T(2π) .* T.(collect(fftfreq(N_grid[3], 1/r_res[3])))

    # Shift to center.
    qxs .= fftshift(qxs)
    qys .= fftshift(qys)
    qzs .= fftshift(qzs)

    # Create the integration grids.
    xs = collect(range(-r_lim[1], step=r_res[1], length=N_grid[1]))
    ys = collect(range(-r_lim[2], step=r_res[2], length=N_grid[2]))
    zs = collect(range(-r_lim[3], step=r_res[3], length=N_grid[3]))

    # Compute the real space integral using Simpson's rule.
    transition_density_sq = Float64.(abs2.(transition_density))

    # Integrate over z first.
    z_problem = SampledIntegralProblem(transition_density_sq, Float64.(zs); dim=3)
    rho_yz = solve(z_problem, SimpsonsRule()).u

    # Then over y.
    y_problem = SampledIntegralProblem(rho_yz, Float64.(ys); dim=2)
    rho_x = solve(y_problem, SimpsonsRule()).u

    # Finally over x.
    x_problem = SampledIntegralProblem(rho_x, Float64.(xs); dim=1)
    rho_integral = solve(x_problem, SimpsonsRule()).u

    # Compute the momentum space integral in the same manner.
    momentum_data_sq = Float64.(abs2.(form_factor))

    # Integrate over z first.
    z_problem = SampledIntegralProblem(momentum_data_sq, Float64.(qzs); dim=3)
    momentum_yz = solve(z_problem, SimpsonsRule()).u

    # Then over y.
    y_problem = SampledIntegralProblem(momentum_yz, Float64.(qys); dim=2)
    momentum_x = solve(y_problem, SimpsonsRule()).u

    # Finally over x.
    x_problem = SampledIntegralProblem(momentum_x, Float64.(qxs); dim=1)
    momentum_integral = solve(x_problem, SimpsonsRule()).u

    println("Real space integral: ", rho_integral)
    println("Momentum space integral/(2π)^3: ", momentum_integral / ((2π)^3))

    return nothing
end

end
