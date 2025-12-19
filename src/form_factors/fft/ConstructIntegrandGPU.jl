module ConstructTransitionDensityGPU

using LinearAlgebra
using CUDA

using ..ReadBasisSet: MoleculeData
using ..ConstructTransitionDensity: construct_spatial_grid

include("../../utils/FastPowers.jl")
using .FastPowers: pow_int

export construct_transition_density_gpu

const THREADS_PER_BLOCK = 512

function compute_orbitals_kernel!(
        orbital_values::CuDeviceArray{T, 2},
        xs::CuDeviceVector{T},
        ys::CuDeviceVector{T},
        zs::CuDeviceVector{T},
        cx::CuDeviceVector{T},
        cy::CuDeviceVector{T},
        cz::CuDeviceVector{T},
        coeffs::CuDeviceVector{T},
        invwidths::CuDeviceVector{T},
        cartesian_a::CuDeviceVector{Int},
        cartesian_b::CuDeviceVector{Int},
        cartesian_c::CuDeviceVector{Int},
        cartesian_prefactor::CuDeviceVector{T},
        primitive_term_offsets::CuDeviceVector{Int},
        primitive_term_indices::CuDeviceVector{Int},
        primitive_to_orbital::CuDeviceVector{Int},
        n_primitives::Int32,
        n_orbitals::Int32,
        Nx::Int32,
        Ny::Int32,
        z_offset::Int32,
        total_points::Int64
    ) where {T}
    """
    A CUDA kernel to compute orbital values for a chunk of grid points. Each thread computes
    the orbital values for one spatial grid point by summing over all primitives.

    # Arguments:
    - orbital_values::CuDeviceArray{T, 2}: The output array to store orbital values with shape (n_orbitals, chunk_points).
    - xs::CuDeviceVector{T}: The x grid points.
    - ys::CuDeviceVector{T}: The y grid points.
    - zs::CuDeviceVector{T}: The z grid points.
    - cx::CuDeviceVector{T}: The x coordinates of primitive centers.
    - cy::CuDeviceVector{T}: The y coordinates of primitive centers.
    - cz::CuDeviceVector{T}: The z coordinates of primitive centers.
    - coeffs::CuDeviceVector{T}: The normalization coefficients for each primitive.
    - invwidths::CuDeviceVector{T}: The inverse widths (1 / (2 * σ^2)) for each primitive.
    - cartesian_a::CuDeviceVector{Int}: The x exponents for Cartesian terms.
    - cartesian_b::CuDeviceVector{Int}: The y exponents for Cartesian terms.
    - cartesian_c::CuDeviceVector{Int}: The z exponents for Cartesian terms.
    - cartesian_prefactor::CuDeviceVector{T}: The prefactors for Cartesian terms.
    - primitive_term_offsets::CuDeviceVector{Int}: Offsets into the term indices for each primitive.
    - primitive_term_indices::CuDeviceVector{Int}: The indices of Cartesian terms for each primitive.
    - primitive_to_orbital::CuDeviceVector{Int}: Mapping from primitive to orbital index.
    - n_primitives::Int32: The number of primitives.
    - n_orbitals::Int32: The number of orbitals.
    - Nx::Int32: The number of x grid points.
    - Ny::Int32: The number of y grid points.
    - z_offset::Int32: The z-index offset for this chunk (0-based).
    - total_points::Int64: The total number of points in this chunk.
    """

    # Get the global thread index and stride.
    idx = Int64((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int64(gridDim().x * blockDim().x)

    @inbounds while idx <= total_points
        # Decode the 1D index into (i, j, k) coordinates within the chunk.
        idx0 = idx - 1
        n_xy = Nx * Ny
        kk_local = idx0 ÷ n_xy + 1  # Local k index within chunk.
        rem_xy = idx0 - (kk_local - 1) * n_xy
        jj = rem_xy ÷ Nx + 1
        ii = rem_xy - (jj - 1) * Nx + 1

        # Compute the global k index by adding the offset.
        kk = kk_local + z_offset

        x = xs[ii]
        y = ys[jj]
        z = zs[kk]

        # Initialise the orbital values for this grid point.
        @inbounds for o in 1:n_orbitals
            orbital_values[o, idx] = zero(T)
        end

        # Sum over all primitives.
        @inbounds for p in 1:n_primitives
            dx = x - cx[p]
            dy = y - cy[p]
            dz = z - cz[p]
            rsq = dx*dx + dy*dy + dz*dz

            exponent = -rsq * invwidths[p]
            # Skip negligible contributions.
            if exponent < -20
                continue
            end

            # Compute the polynomial part.
            poly = zero(T)
            start_idx = primitive_term_offsets[p]
            end_idx = primitive_term_offsets[p + 1] - 1
            for t in start_idx:end_idx
                term_idx = primitive_term_indices[t]
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

            val = coeffs[p] * poly * exp(exponent)
            orbital_idx = primitive_to_orbital[p]
            orbital_values[orbital_idx, idx] += val
        end

        idx += stride
    end
    return
end

function compute_transition_density_kernel!(
        output::CuDeviceArray{T, 1},
        orbitals::CuDeviceArray{T, 2},
        temp::CuDeviceArray{T, 2},
        n_orbitals::Int32,
        total_points::Int64
    ) where {T}
    """
    A CUDA kernel to compute the transition density.

    # Arguments:
    - output::CuDeviceArray{T, 1}: Output array for transition density with shape (chunk_points,).
    - orbitals::CuDeviceArray{T, 2}: Orbital values with shape (n_orbitals, chunk_points).
    - temp::CuDeviceArray{T, 2}: Temp buffer (T*ϕ) with shape (n_orbitals, chunk_points).
    - n_orbitals::Int32: Number of orbitals.
    - total_points::Int64: Total number of grid points in this chunk.
    """

    # Get the global thread index and stride.
    idx = Int64((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    stride = Int64(gridDim().x * blockDim().x)

    @inbounds while idx <= total_points
        # Compute ρ[idx] = ∑_i temp[i, idx] * orbitals[i, idx]
        rho = zero(T)
        for i in 1:n_orbitals
            rho += temp[i, idx] * orbitals[i, idx]
        end
        output[idx] = rho
        idx += stride
    end
    return
end

function construct_transition_density_gpu(
        mol::MoleculeData{T},
        transition_matrices::Vector{Matrix{T}},
        qx_grid::Vector{T},
        qy_grid::Vector{T},
        qz_grid::Vector{T}
    )::Tuple{CuArray{T, 4}, Vector{T}} where {T<:AbstractFloat}
    """
    Construct the transition density ϕ_f(r)T_{fi}ϕ_i(r) for one or more transitions using the GPU.

            This implementation processes the grid in memory-safe z-chunks.
            Each slice launches a single kernel to compute orbital values, then uses GPU matrix operations for transitions.

    # Arguments:
    - mol::MoleculeData{T}: The molecular data structure containing basis set and coordinates.
    - transition_matrices::Vector{Matrix{T}}: Vector of transition matrices T_{fi}.
    - qx_grid::Vector{T}: The qx grid in momentum space, in keV.
    - qy_grid::Vector{T}: The qy grid in momentum space, in keV.
    - qz_grid::Vector{T}: The qz grid in momentum space, in keV.

    # Returns:
    - transition_densities::CuArray{T, 4}: Transition densities on GPU with shape (n_transitions, n_x, n_y, n_z).
    - r_lim::Vector{T}: The limits of the grid in real space, in Angstroms.
    """

    xs, ys, zs, r_lim, N_grid = construct_spatial_grid(qx_grid, qy_grid, qz_grid)

    n_primitives = mol.n_primitives
    n_orbitals = mol.n_orbitals
    n_transitions = length(transition_matrices)

    # Determine optimal z_chunk_size based on available GPU memory.
    # We need to fit orbital_chunk_gpu and temp_chunk_gpu in memory.
    # Memory per chunk: 2 × n_orbitals × N_x × N_y × z_chunk_size × sizeof(T) bytes.
    available_memory = CUDA.available_memory()
    bytes_per_point = 2 * n_orbitals * sizeof(T)
    n_xy = N_grid[1] * N_grid[2]
    bytes_per_z_slice = bytes_per_point * n_xy

    # Target using ~25% of available memory for the chunk buffers.
    target_memory = T(0.25) * available_memory
    max_z_chunk_size = max(1, floor(Int, target_memory / bytes_per_z_slice))

    z_chunk_size = min(max_z_chunk_size, N_grid[3])

    # For very small grids or low memory situations, ensure at least z_chunk_size=1 works.
    z_chunk_size = max(1, z_chunk_size)
    primitive_to_atom = mol.primitive_to_atom
    primitive_to_orbital = mol.primitive_to_orbital

    centres = mol.atom_coordinates[primitive_to_atom, :]
    cx = centres[:,1]; cy = centres[:,2]; cz = centres[:,3]
    coeffs = mol.normalised_coefficients
    invwidths = T(1.0) ./ (T(2.0) .* mol.widths .^2)

    cartesian_a = mol.cartesian_a
    cartesian_b = mol.cartesian_b
    cartesian_c = mol.cartesian_c
    cartesian_prefactor = mol.cartesian_prefactor
    cartesian_term_to_primitive = mol.cartesian_term_to_primitive

    primitive_term_offsets = zeros(Int, n_primitives + 1)
    primitive_term_indices = Int[]
    for p in 1:n_primitives
        primitive_term_offsets[p] = length(primitive_term_indices) + 1
        for (term_idx, primitive_idx) in enumerate(cartesian_term_to_primitive)
            primitive_idx == p && push!(primitive_term_indices, term_idx)
        end
    end
    primitive_term_offsets[end] = length(primitive_term_indices) + 1

    # Transfer grid points and primitive data to GPU.
    xs_gpu = CuArray(xs)
    ys_gpu = CuArray(ys)
    zs_gpu = CuArray(zs)
    cx_gpu = CuArray(cx)
    cy_gpu = CuArray(cy)
    cz_gpu = CuArray(cz)
    coeffs_gpu = CuArray(coeffs)
    invwidths_gpu = CuArray(invwidths)
    cartesian_a_gpu = CuArray(cartesian_a)
    cartesian_b_gpu = CuArray(cartesian_b)
    cartesian_c_gpu = CuArray(cartesian_c)
    cartesian_prefactor_gpu = CuArray(cartesian_prefactor)
    primitive_term_offsets_gpu = CuArray(primitive_term_offsets)
    primitive_term_indices_gpu = CuArray(primitive_term_indices)
    primitive_to_orbital_gpu = CuArray(primitive_to_orbital)

    # Transfer transition matrices to GPU.
    transition_matrices_gpu = [CuArray(tm) for tm in transition_matrices]

    # Allocate output for transition densities on GPU.
    transition_densities_gpu = CUDA.zeros(T, n_transitions, N_grid[1], N_grid[2], N_grid[3])

    # Process the grid in z-slice chunks to balance memory and kernel launches.
    n_xy = N_grid[1] * N_grid[2]
    n_chunks = cld(N_grid[3], z_chunk_size)

    # Allocate reusable buffers for chunk processing.
    max_chunk_points = n_xy * z_chunk_size
    orbital_chunk_gpu = CUDA.zeros(T, n_orbitals, max_chunk_points)
    temp_chunk_gpu = CUDA.zeros(T, n_orbitals, max_chunk_points)
    transition_chunk_gpu = CUDA.zeros(T, max_chunk_points)  # Reusable buffer for transition density.

    for chunk_idx in 1:n_chunks
        # Determine the z-range for this chunk.
        z_start = (chunk_idx - 1) * z_chunk_size + 1
        z_end = min(chunk_idx * z_chunk_size, N_grid[3])
        chunk_nz = z_end - z_start + 1
        chunk_points = n_xy * chunk_nz

        # Compute the offset for this chunk in the full grid.
        offset = (z_start - 1) * n_xy

        # Launch the kernel to compute orbital values for this chunk.
        blocks = cld(chunk_points, THREADS_PER_BLOCK)
        @cuda threads=THREADS_PER_BLOCK blocks=blocks compute_orbitals_kernel!(
            orbital_chunk_gpu,
            xs_gpu,
            ys_gpu,
            zs_gpu,
            cx_gpu,
            cy_gpu,
            cz_gpu,
            coeffs_gpu,
            invwidths_gpu,
            cartesian_a_gpu,
            cartesian_b_gpu,
            cartesian_c_gpu,
            cartesian_prefactor_gpu,
            primitive_term_offsets_gpu,
            primitive_term_indices_gpu,
            primitive_to_orbital_gpu,
            Int32(n_primitives),
            Int32(n_orbitals),
            Int32(N_grid[1]),
            Int32(N_grid[2]),
            Int32(z_start - 1),  # 0-based z offset
            Int64(chunk_points)
        )
        CUDA.synchronize()

        # Process each transition for this chunk.
        for t_idx in 1:n_transitions
            # Compute T * ϕ for this chunk.
            chunk_view = view(orbital_chunk_gpu, :, 1:chunk_points)
            temp_view = view(temp_chunk_gpu, :, 1:chunk_points)
            mul!(temp_view, transition_matrices_gpu[t_idx], chunk_view)
            CUDA.synchronize()

            # Compute transition density: ρ = ∑_i (T*ϕ)_i * ϕ_i.
            transition_chunk_view = view(transition_chunk_gpu, 1:chunk_points)
            blocks_reduction = cld(chunk_points, THREADS_PER_BLOCK)
            @cuda threads=THREADS_PER_BLOCK blocks=blocks_reduction compute_transition_density_kernel!(
                transition_chunk_view,
                chunk_view,
                temp_view,
                Int32(n_orbitals),
                Int64(chunk_points)
            )
            CUDA.synchronize()

            # Store the result in an output array, reshaping to 3D for indexing.
            output_flat = reshape(view(transition_densities_gpu, t_idx, :, :, :), :)
            copyto!(view(output_flat, (offset + 1):(offset + chunk_points)), transition_chunk_view)
            CUDA.synchronize()
        end
    end

    # Apply the spin-degeneracy factor.
    transition_densities_gpu .*= sqrt(T(2))

    return transition_densities_gpu, r_lim
end

end
