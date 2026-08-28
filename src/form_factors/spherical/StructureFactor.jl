# This module builds the energy axis of the structure function, and the broadened delta that puts
# each crystal state onto it.

module StructureFactor

using StaticArrays
using Base.Threads

using ..CrystalLattice: CrystalLatticeData, fold_to_bz
using ..Ewald: EwaldLongRangeData
using ..BlochHamiltonian: CrystalExcitationBasis, BlochEigensystem, CrystalCouplings,
                          solve_bloch_energies!
using ..CrystalSymmetry: Stars
using ...ThreadChunks: chunk_count, chunk_range

export EnergyGrid, build_energy_grid, accumulate_delta!, incoherent_structure_factor,
       integrate_energy, trim_energy_axis, describe_energy_grid, kernel_shape

# We use a mollifier in place of the delta function, with width either side of the band σ_E = 3 ΔE.
const MOLLIFIER_SUPPORT_BINS = 3

struct EnergyGrid{T<:AbstractFloat}
    energies::Vector{T}   # The E axis, in eV.
    sigma::T              # σ_E, where the kernel reaches zero. Nothing lies beyond it.
    step::T               # ΔE between neighbouring bins.
    half_window::Int      # Bins either side of a band energy that the kernel reaches.
end


@inline function kernel_shape(grid::EnergyGrid{T}, offset::T) where {T<:AbstractFloat}
    """
    Evaluate the unnormalised mollifier at a distance offset from its centre,

        φ(x) = exp[-1 / (1 - x²)]  for |x| < 1,   0 otherwise,     x = offset / σ_E.

    # Arguments:
    - grid::EnergyGrid{T}: The energy axis, carrying σ_E.
    - offset::T: E_bin - E_Ψ, in eV.

    # Returns:
    - T: The unnormalised weight.
    """

    scaled = offset / grid.sigma
    squared = scaled * scaled

    return squared < 1 ? exp(-inv(1 - squared)) : zero(T)
end


function band_energy_range(
        basis::CrystalExcitationBasis{T},
        lattice::CrystalLatticeData{T},
        q_grid_invA::Vector{T},
        stars::Stars{T},
        couplings::Union{Nothing, CrystalCouplings{T}},
        long_range::Union{Nothing, EwaldLongRangeData{T}};
        star_stride::Int = 8,
        q_stride::Int = 4,
    ) where {T<:AbstractFloat}
    """
    Find the range of E_Ψ(k) over the wavevectors the form factor will visit.

    This samples the k of the coherent loop, k = fold(q n̂) per star representative, but only every
    star_stride-th star and q_stride-th q.

    Because it is a sample and could underestimate the range, the range is widened by the largest change in any band energy between
    neighbouring sampled points along a ray (fixed angle). These are skipped later.

    Anything missed still is flagged.

    # Arguments:
    - basis::CrystalExcitationBasis{T}: The localised excitation basis.
    - lattice::CrystalLatticeData{T}: The crystal lattice.
    - q_grid_invA::Vector{T}: The |q| grid, in inverse Å.
    - stars::Stars{T}: The direction stars.
    - couplings::Union{Nothing, CrystalCouplings{T}}: The J_{λλ'}(ΔR), or nothing.
    - long_range::Union{Nothing, EwaldLongRangeData{T}}: The Ewald long-range data, or nothing.
    - star_stride::Int: Sample every this many star representatives (default: 8).
    - q_stride::Int: Sample every this many q points (default: 4).

    # Returns:
    - Tuple{T, T, T}: The smallest and largest sampled band energy in eV, and the margin the caller
      should widen them by.
    """

    # Get the dimensions.
    n_stars = length(stars.irreducible_directions)
    n_cells = couplings === nothing ? 0 : length(couplings.cell_vectors)
    n_lambda = length(basis.energies)

    # Sample the wavevectors and q.
    sampled_stars = 1:star_stride:n_stars
    sampled_q = 1:q_stride:length(q_grid_invA)
    n_sampled = length(sampled_stars)
    n_chunks = chunk_count(n_sampled, nthreads())

    chunk_low = fill(T(Inf), n_chunks)
    chunk_high = fill(T(-Inf), n_chunks)
    chunk_step = zeros(T, n_chunks)

    # Thread over stars.
    @sync for chunk in 1:n_chunks
        Threads.@spawn begin
            # Build the eigensystem for this star.
            eigensystem = BlochEigensystem(basis, long_range; n_cells = n_cells, vecs = false)
            previous = Vector{T}(undef, n_lambda)
            # Keep track of the bounds for this chunk.
            low = T(Inf)
            high = T(-Inf)
            largest_step = zero(T)

            for sample_idx in chunk_range(chunk, n_chunks, n_sampled)
                representative = stars.irreducible_directions[sampled_stars[sample_idx]]

                for (position, q_idx) in enumerate(sampled_q)
                    # Solve the eigensystem for this star representative direction and q, which sets the magnitude.
                    k_vector, _ = fold_to_bz(lattice, q_grid_invA[q_idx] * representative)
                    solve_bloch_energies!(eigensystem, basis, k_vector, couplings, long_range)

                    # Update the bounds.
                    low = min(low, eigensystem.energies[1])
                    high = max(high, eigensystem.energies[end])

                    # How far any band moved over one sampling step, which sets the margin.
                    if position > 1
                        for state in 1:n_lambda
                            largest_step = max(largest_step,
                                               abs(eigensystem.energies[state] - previous[state]))
                        end
                    end
                    copyto!(previous, eigensystem.energies)
                end
            end

            chunk_low[chunk] = low
            chunk_high[chunk] = high
            chunk_step[chunk] = largest_step
        end
    end

    # Set the margin to half the biggest per point shift we saw above.
    margin = maximum(chunk_step) / 2

    return minimum(chunk_low), maximum(chunk_high), margin
end


function build_energy_grid(
        energy_low::T,
        energy_high::T,
        step::T;
        maximum_points::Int = 200_000,
    ) where {T<:AbstractFloat}
    """
    Build the energy axis at the requested bin width, and the mollifier that goes on it.

    # Arguments:
    - energy_low::T: The smallest band energy, in eV.
    - energy_high::T: The largest band energy, in eV.
    - step::T: ΔE, the bin width, in eV.
    - maximum_points::Int: Refuse to build an axis longer than this, since the cost of everything
      downstream is linear in it (default: 200000).

    # Returns:
    - EnergyGrid{T}: The axis and its kernel.
    """

    # Check the grid is sensible.
    step > 0 || error("ΔE must be positive, got $(step) eV.")
    span = energy_high - energy_low
    span > 0 || error("The band range is empty: E_min = $(energy_low), E_max = $(energy_high).")

    # Set σ_E, the width of the mollifier.
    sigma = T(MOLLIFIER_SUPPORT_BINS) * step
    # Set the number of energy points, as span + room for the mollifier width at either end.
    n_energy = ceil(Int, (span + 2 * sigma) / step) + 1

    n_energy <= maximum_points ||
        error("ΔE = $(step) eV needs $(n_energy) energy points to cover a band range of " *
              "$(round(span, digits = 4)) eV, over the limit of $(maximum_points). Use a larger ΔE.")

    # Build the energy grid.
    energies = collect(range(energy_low - sigma; step = step, length = n_energy))

    return EnergyGrid{T}(energies, sigma, step, MOLLIFIER_SUPPORT_BINS)
end


function describe_energy_grid(grid::EnergyGrid{T}) where {T<:AbstractFloat}
    """
    Render the energy grid as a line for the terminal.

    # Arguments:
    - grid::EnergyGrid{T}: The grid.

    # Returns:
    - String: The description.
    """

    return string(
        "Structure function energy grid: ", length(grid.energies), " points over [",
        round(first(grid.energies), digits = 4), ", ", round(last(grid.energies), digits = 4),
        "] eV, ΔE = ", round(1000 * grid.step, digits = 3), " meV.\n",
        "  Mollifier delta, σ_E = ±", round(1000 * grid.sigma, digits = 3),
        " meV over ±", grid.half_window, " bins, and exactly zero outside.",
    )
end


@inline function accumulate_delta!(
        buffer::AbstractArray{T, 4},
        occupancy::Matrix{Bool},
        grid::EnergyGrid{T},
        energy::T,
        magnitude::T,
        q_local::Int,
        theta_idx::Int,
        phi_idx::Int,
    ) where {T<:AbstractFloat}
    """
    Spread the form factor over the energy axis to get the structure function, using the mollifier.
    This is approximately:
        
            f_Ψ(q, E) ≃ f_Ψ(q) * δ(E - E_Ψ(q)),

    with the mollifier in place of the delta function.

    # Arguments:
    - buffer::AbstractArray{T, 4}: The accumulator, dimensions (n_energy, n_q_block, n_theta, n_phi).
    - occupancy::Matrix{Bool}: Which (bin, q) ever receive anything, dimensions (n_energy,
      n_q_block), marked here so the projection can skip the rest. One array per task, merged later.
    - grid::EnergyGrid{T}: The energy axis and its kernel.
    - energy::T: E_Ψ(k) for this state, in eV.
    - magnitude::T: |f_Ψ(q)|².
    - q_local::Int: The q index within the block.
    - theta_idx::Int: The θ index.
    - phi_idx::Int: The ϕ index.

    # Returns:
    - Bool: Whether the state fell outside the axis, so the caller can count it. The energy range is
      sampled rather than swept, so this is the check that an underestimate is loud rather than
      silent: the weight is still deposited, at the nearest edge, but at the wrong energy.
    """

    n_energy = length(grid.energies)

    # The grid is uniform, so the nearest bin is arithmetic rather than a search.
    raw_centre = round(Int, (energy - grid.energies[1]) / grid.step) + 1
    centre = clamp(raw_centre, 1, n_energy)
    outside = raw_centre != centre
    # Find the range of bins we need to look at.
    first_bin = max(1, centre - grid.half_window)
    last_bin = min(n_energy, centre + grid.half_window)

    total = zero(T)
    # Add the mollifier at each energy.
    @inbounds for bin in first_bin:last_bin
        total += kernel_shape(grid, grid.energies[bin] - energy)
    end

    # Handle the cases when our energy grid is too small, which should never happen.
    if total <= 0
        buffer[centre, q_local, theta_idx, phi_idx] += magnitude / grid.step
        occupancy[centre, q_local] = true
        return outside
    end

    # Now include the |f|^2 part, and the normalisation factor that ensures the sum rule holds.
    weight = magnitude / (total * grid.step)
    @inbounds for bin in first_bin:last_bin
        occupancy[bin, q_local] = true
        buffer[bin, q_local, theta_idx, phi_idx] +=
            weight * kernel_shape(grid, grid.energies[bin] - energy)
    end

    return outside
end


function incoherent_structure_factor(
        f_lm::Array{T, 3},
        energies::Vector{T},
        grid::EnergyGrid{T},
    ) where {T<:AbstractFloat}
    """
    Spread the incoherent f²_{ℓm}(q) onto the energy axis.

    Without the Bloch problem each transition keeps one energy for every q, so there is no angular
    integral left to do and the structure function is just the existing coefficients times one fixed
    kernel per transition.

    # Arguments:
    - f_lm::Array{T, 3}: The coefficients, dimensions (n_transitions, n_q, n_keys).
    - energies::Vector{T}: The transition energies in eV, one per transition.
    - grid::EnergyGrid{T}: The energy axis and its kernel.

    # Returns:
    - Array{T, 3}: f²_{ℓm}(q, E), dimensions (n_energy, n_q, n_keys).
    """

    # Get the dimensions.
    n_transitions, n_q, n_keys = size(f_lm)
    n_energy = length(grid.energies)
    length(energies) == n_transitions ||
        error("Got $(length(energies)) energies for $(n_transitions) transitions.")

    structure_factor = zeros(T, n_energy, n_q, n_keys)
    kernel = Vector{T}(undef, n_energy)

    for transition in 1:n_transitions
        # One kernel per transition, since its energy does not move with q.
        total = zero(T)
        @inbounds for bin in 1:n_energy
            offset = grid.energies[bin] - energies[transition]
            kernel[bin] = abs(offset) <= grid.half_window * grid.step ?
                          kernel_shape(grid, offset) : zero(T)
            total += kernel[bin]
        end
        total > 0 || continue
        # Normalise the kernel.
        kernel ./= total * grid.step

        # Spread the energies.
        @inbounds for key in 1:n_keys, q_idx in 1:n_q
            value = f_lm[transition, q_idx, key]
            value == 0 && continue
            for bin in 1:n_energy
                structure_factor[bin, q_idx, key] += value * kernel[bin]
            end
        end
    end

    return structure_factor
end


function trim_energy_axis(
        structure_factor::Array{T, 3},
        grid::EnergyGrid{T},
    ) where {T<:AbstractFloat}
    """
    Drop the bins at either end of the energy axis that nothing ever landed in.
    This removes the extra padding we put in before, to save disk space later.

    # Arguments:
    - structure_factor::Array{T, 3}: f²_{ℓm}(q, E), dimensions (n_energy, n_q, n_keys).
    - grid::EnergyGrid{T}: The axis it sits on.

    # Returns:
    - Array{T, 3}: The trimmed structure function.
    - EnergyGrid{T}: The matching trimmed axis.
    """

    # Get the dimensions.
    n_energy, n_q, n_keys = size(structure_factor)

    occupied = falses(n_energy)
    # Check which bins are occupied.
    @inbounds for key in 1:n_keys, q_idx in 1:n_q, bin in 1:n_energy
        structure_factor[bin, q_idx, key] == 0 || (occupied[bin] = true)
    end

    # Find the first and last occupied bin indices, and trim.
    first_bin = findfirst(occupied)
    first_bin === nothing && return structure_factor, grid
    last_bin = findlast(occupied)
    (first_bin == 1 && last_bin == n_energy) && return structure_factor, grid

    kept = first_bin:last_bin

    return structure_factor[kept, :, :],
           EnergyGrid{T}(grid.energies[kept], grid.sigma, grid.step, grid.half_window)
end


function integrate_energy(
        structure_factor::Array{T, 3},
        grid::EnergyGrid{T},
    ) where {T<:AbstractFloat}
    """
    Integrate the structure function over E, which must give back f²_{ℓm}(q).

    The kernel is normalised, so this is the sum rule that says every state deposited its magnitude
    once and only once.

    # Arguments:
    - structure_factor::Array{T, 3}: f²_{ℓm}(q, E), dimensions (n_energy, n_q, n_keys).
    - grid::EnergyGrid{T}: The energy axis.

    # Returns:
    - Matrix{T}: ∫dE f²_{ℓm}(q, E), dimensions (n_q, n_keys).
    """

    # Get the dimensions.
    n_energy, n_q, n_keys = size(structure_factor)
    integral = zeros(T, n_q, n_keys)

    # The thing saved is in units of inverse dE, so multiply by dE to get the integral.
    @inbounds for key in 1:n_keys, q_idx in 1:n_q
        total = zero(T)
        for bin in 1:n_energy
            total += structure_factor[bin, q_idx, key]
        end
        integral[q_idx, key] = total * grid.step
    end

    return integral
end

end
