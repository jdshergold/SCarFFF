module ConstructTransitionDensity

using LinearAlgebra
using Base.Threads
using Printf

using ..ReadBasisSet: MoleculeData

include("../../utils/FastPowers.jl")
using .FastPowers: pow_int

export construct_transition_density

const KEV_TO_INV_ANGSTROM = 1/1.973269803 # Conversion factor from keV to inverse Angstroms.

function construct_spatial_grid(
        qx_grid::Vector{T},
        qy_grid::Vector{T},
        qz_grid::Vector{T}
    )::Tuple{Vector{T}, Vector{T}, Vector{T}, Vector{T}, Vector{Int}} where {T<:AbstractFloat}
    """
    Construct the real-space grid parameters from momentum-space grid.
    
    # Arguments:
    - qx_grid::Vector{T}: The qx grid in momentum space, in keV.
    - qy_grid::Vector{T}: The qy grid in momentum space, in keV.
    - qz_grid::Vector{T}: The qz grid in momentum space, in keV.

    # Returns:
    - xs::Vector{T}: The x-coordinates of the spatial grid in Angstroms.
    - ys::Vector{T}: The y-coordinates of the spatial grid in Angstroms.
    - zs::Vector{T}: The z-coordinates of the spatial grid in Angstroms.
    - r_lim::Vector{T}: The limits of the grid in real space, in Angstroms.
    - N_grid::Vector{Int}: The number of grid points [Nx, Ny, Nz].
    """
    
    # Extract the grid parameters.
    N_grid = [length(qx_grid), length(qy_grid), length(qz_grid)]
    
    # Compute the grid spacing from the provided grids.
    q_res = [T(abs(qx_grid[2] - qx_grid[1])),
             T(abs(qy_grid[2] - qy_grid[1])),
             T(abs(qz_grid[2] - qz_grid[1]))
    ]
    
    # Find the corresponding real-space grid parameters.
    q_res_invA = q_res .* T(KEV_TO_INV_ANGSTROM)
    r_res = T(2π) ./ (N_grid .* q_res_invA) # Δx = 2π / (N Δq).
    r_lim = T(0.5) .* r_res .* N_grid

    xs = collect(range(-r_lim[1], step=r_res[1], length=N_grid[1]))
    ys = collect(range(-r_lim[2], step=r_res[2], length=N_grid[2]))
    zs = collect(range(-r_lim[3], step=r_res[3], length=N_grid[3]))
    
    return xs, ys, zs, r_lim, N_grid
end

function construct_transition_density(
        mol::MoleculeData{T},
        transition_matrices::Vector{Matrix{T}},
        qx_grid::Vector{T},
        qy_grid::Vector{T},
        qz_grid::Vector{T}
    )::Tuple{Array{T, 4}, Vector{T}} where {T<:AbstractFloat}
    """
    Construct the transition density ϕ_f(r)T_{fi}ϕ_i(r) for one or more transitions.

    Here ϕ_{i}(r - r_i) is the i-th atomic orbital at position r_i, given by:

        ϕ_{i}(r) = ∑_p c_{p}^{i} ∑_{abc} pref_{abc} (x-x_i)^a (y-y_i)^b (z-z_i)^c e^{-α_{p}^{i} r^2},

    where p is the primitive index and abc are Cartesian expansion terms.

    # Arguments:
    - mol::MoleculeData{T}: The molecular data structure containing basis set and coordinates.
    - transition_matrices::Vector{Matrix{T}}: Vector of transition matrices T_{fi}.
    - qx_grid::Vector{T}: The qx grid in momentum space, in keV.
    - qy_grid::Vector{T}: The qy grid in momentum space, in keV.
    - qz_grid::Vector{T}: The qz grid in momentum space, in keV.

    # Returns:
    - transition_densities::Array{T, 4}: A 4D array representing the transition densities with shape (n_transitions, n_x, n_y, n_z).
    - r_lim::Vector{T}: The limits of the grid in real space, in Angstroms.
    """

    # Construct the real-space grid from the momentum-space grid.
    xs, ys, zs, r_lim, N_grid = construct_spatial_grid(qx_grid, qy_grid, qz_grid)

    # Compute the transition density on the spatial grid.
    n_primitives = mol.n_primitives
    n_orbitals = mol.n_orbitals
    n_transitions = length(transition_matrices)
    primitive_to_atom = mol.primitive_to_atom
    primitive_to_orbital = mol.primitive_to_orbital

    # Build an array that directly maps primitive indices to their respective centers.
    # Work with 1D arrays for speed.
    centres = mol.atom_coordinates[primitive_to_atom, :]
    cx = centres[:,1]; cy = centres[:,2]; cz = centres[:,3]
    coeffs = mol.normalised_coefficients
    invwidths = 1.0 ./ (2.0 .* mol.widths .^2) # Do this division only once for speed, division is expensive.

    # Extract the Cartesian terms.
    cartesian_a = mol.cartesian_a
    cartesian_b = mol.cartesian_b
    cartesian_c = mol.cartesian_c
    cartesian_prefactor = mol.cartesian_prefactor
    cartesian_term_to_primitive = mol.cartesian_term_to_primitive

    # Build index array mapping primitives to their Cartesian terms
    primitive_term_indices = [Int[] for _ in 1:n_primitives]
    for (term_idx, primitive_idx) in enumerate(cartesian_term_to_primitive)
        push!(primitive_term_indices[primitive_idx], term_idx)
    end

    # Preallocate the transition_densities array.
    transition_densities = zeros(T, n_transitions, N_grid[1], N_grid[2], N_grid[3])

    # Prellocate buffers for speed, and one to each thread to avoid races.
    n_threads = Threads.nthreads()
    primitive_pool = [zeros(T, n_primitives) for _ in 1:n_threads] # For storing primitive contributions.
    orbital_pool = [zeros(T, n_orbitals) for _ in 1:n_threads] # For storing orbital contributions.
    temp_pool = [zeros(T, n_orbitals, n_transitions) for _ in 1:n_threads] # For storing T_fi * ϕ_i for all transitions.

    # Compute the transition density one grid point at a time. Each thread works on a different z-slice.
    @threads for kk in 1:N_grid[3]
        z = zs[kk]

        # Allocate thread-local buffers.
        tid = threadid()
        primitive_local = primitive_pool[tid]
        orbital_local = orbital_pool[tid]
        temp_local = temp_pool[tid]

        for jj in 1:N_grid[2]
            y = ys[jj]
            for ii in 1:N_grid[1]
                x = xs[ii]

                # Compute each primitive contribution at the current grid point (x, y, z).
                @inbounds for p in 1:n_primitives
                    dx = x - cx[p]
                    dy = y - cy[p]
                    dz = z - cz[p]
                    rsq = dx*dx + dy*dy + dz*dz

                    # Skip the expensive exponential if the primitive contribution is negligible.
                    exponent = -rsq * invwidths[p]
                    if exponent < -20 # exp(-20) ~ 4.85e-10, so negligible contribution.
                        primitive_local[p] = 0.0
                        continue
                    end

                    # Evaluate Cartesian polynomial expansion
                    poly = 0.0
                    for term_idx in primitive_term_indices[p]
                        pref = cartesian_prefactor[term_idx]
                        a = cartesian_a[term_idx]
                        b = cartesian_b[term_idx]
                        c = cartesian_c[term_idx]
                        term_val = pref
                        a != 0 && (term_val *= pow_int(dx, a))
                        b != 0 && (term_val *= pow_int(dy, b))
                        c != 0 && (term_val *= pow_int(dz, c))
                        poly += term_val
                    end

                    primitive_local[p] = coeffs[p] * poly * exp(exponent)
                end

                # If all primitives are zero, so too will be all orbitals, so skip accumulation and matmul.
                if all(primitive_local .== 0.0)
                    for t in 1:n_transitions
                        transition_densities[t, ii, jj, kk] = zero(T)
                    end
                    continue
                end

                # Accumulate primitives into the correct orbitals.
                fill!(orbital_local, 0.0)
                @inbounds for p in 1:n_primitives
                    orbital_local[primitive_to_orbital[p]] += primitive_local[p]
                end

                # Compute the transition density at the current grid point for ALL transitions.
                # This reuses the expensive orbital evaluation across all transitions.
                for t in 1:n_transitions
                    mul!(view(temp_local, :, t), transition_matrices[t], orbital_local)  # temp[:, t] = T_fi * ϕ_i.
                    transition_densities[t, ii, jj, kk] = T(dot(orbital_local, view(temp_local, :, t)))
                end
            end
        end
    end

    # Finally multiply by sqrt(2) for spin degeneracy.
    transition_densities .*= sqrt(T(2))

    return transition_densities, r_lim
end

end
