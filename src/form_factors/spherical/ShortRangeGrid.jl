# Choose the short-range radial grid from the W tensor, density matrices and crystal distances.

using StaticArrays: SVector
using LinearAlgebra: norm

function short_range_envelope(W, sigma_ij, mol, density_matrices, n_transitions)
    """
    Construct the Gaussian envelopes for the electronic FF families F = f_λ, n and Δ:

        B_F(q) = ∑_{σ,n} A^F_{σn} q^n exp(-σ²q²/2),
        A^F_{σn} = 2 (2π)^(3/2) ∑_{i≤j: σ_ij=σ} |w^F_ij| ∑_{λ,μ} |W^n_{ij,λμ}| √((2λ+1)/(4π)).

    Here f_λ is the monomer transition FF, n the ground state electron density and Δ the difference
    density Φ^{t→s} - δ_st n. The weight w^F_ij is the AO density matrix entry T^F_{μν}, with μ and ν
    the orbitals of Cartesian terms i and j: the diagonal entry for i = j, or the sum of the ij and
    ji entries otherwise. For f_λ and Δ, use the largest absolute weight over the requested states
    or state pairs.
    The spherical harmonic is bounded by √((2λ+1)/(4π)); the translation phase has magnitude one.
    Thus B_F contains the Gaussian and polynomial amplitudes, before any Bessel decay is included.

    Only the grouped amplitudes are built here, so no form factors need to be evaluated.

    # Arguments:
    - W::SparseWTensor{T}: The molecular W tensor, after applying the W threshold.
    - sigma_ij::Array{T,2}: The Gaussian product widths, in Å.
    - mol::MoleculeData{T}: The molecular data.
    - density_matrices::Vector{Matrix{T}}: The per-spin AO density matrices, ordered as transitions, n, then Δ.
    - n_transitions::Int: The number of transitions.

    # Returns:
    - NamedTuple: The widths sigma, powers n and amplitudes A, with shape (3, n_moments).
      Also contains r_N = max_I |R_I|, Z = ∑_I Z_I and r_Z = Z / (∑_I Z_I/|R_I|), with lengths in Å.
    """

    T = eltype(sigma_ij)
    # Map each Cartesian pair to its AO density weight, as in the R tensor contraction.
    bins, pair_i, pair_j, weights = ConstructRTensor.build_pair_density_weights(
        W, density_matrices, mol.cartesian_term_to_orbital)
    amplitudes = Dict{Tuple{T, Int}, Vector{T}}()
    for pair in eachindex(bins)
        # One envelope must cover every transition or state pair in each family.
        f_weight = maximum(abs, view(weights, 1:n_transitions, pair))
        g_weight = abs(weights[n_transitions + 1, pair])
        Delta_weight = maximum(abs, view(weights, (n_transitions + 2):size(weights, 1), pair))
        sigma = sigma_ij[pair_i[pair], pair_j[pair]]
        for entry in W.ij_bins[bins[pair]]
            # Sum over λ and μ now, keeping only the Gaussian width and polynomial power.
            A = get!(amplitudes, (sigma, W.n[entry])) do
                zeros(T, 3)
            end
            # Bound |Y_λμ| and include the same spin and Fourier prefactor as the molecular FF.
            angular = sqrt(T(2 * W.lambda[entry] + 1) / T(4π))
            value = T(2 * (2π)^(3 / 2)) * abs(W.W_values[entry]) * angular
            A[1] += f_weight * value
            A[2] += g_weight * value
            A[3] += Delta_weight * value
        end
    end
    # Store the grouped terms in a fixed order for the later tail evaluations.
    moments = sort!(collect(keys(amplitudes)))
    sigma = T[key[1] for key in moments]
    n = Int[key[2] for key in moments]
    A = reduce(hcat, [amplitudes[key] for key in moments])
    # r_N sets the largest molecular phase distance. r_Z estimates the nuclear Bessel amplitude.
    r_N, inverse_radius = zero(T), zero(T)
    for atom in axes(mol.atom_coordinates, 1)
        radius = norm(view(mol.atom_coordinates, atom, :))
        r_N = max(r_N, radius)
        # A nucleus at the origin has j_0(0) = 1, so there is no 1/(qR_I) decay for that nucleus.
        inverse_radius += radius > eps(T) ? T(mol.nuclear_charges[atom]) / radius : T(Inf)
    end
    Z = T(sum(mol.nuclear_charges))
    return (; sigma, n, A, r_N, Z, r_Z = Z / inverse_radius)
end

function short_range_geometry(data, basis, cells, parameters)
    """
    Measure the molecular and crystal distances entering the SR spacing and tail estimates:

        d_{αβ} = τ_i - τ_j - ΔR,
        d_max = max_{SR neighbours} |d_{αβ}|,
        d_min = min_{SR neighbours,I,J} |d_{αβ} + ℛ_α R_I - ℛ_β R_J|,
        P = max_α ∑_{β in SR neighbours} 1/|d_{αβ}|.

    ℛ_α is the full orientation matrix of image α, including any improper rotation, and R_I is the
    position of nucleus I in the monomer frame. d_min is the closest nucleus-to-nucleus distance
    between distinct molecules, including periodic copies. It is used as a representative slow phase
    distance in the tail model. d_max sets the largest intermolecular phase distance for the spacing
    estimate.

    # Arguments:
    - data: The prepared molecular quantities for each conformer.
    - basis::CrystalExcitationBasis{T}: The crystal excitation basis.
    - cells::NeighbourCells{T}: The real-space neighbour cells.
    - parameters::EwaldParameters{T}: The Ewald parameters.

    # Returns:
    - NamedTuple: d_min, d_max, r_N, sigma_max and r_Z in Å, P in inverse Å and total nuclear charge Z.
      Across conformers, use the largest r_N, sigma_max and Z, and the smallest r_Z.
    """

    T = typeof(parameters.eta)
    # Rotate the nuclei into each crystal image's orientation, keeping positions relative to its centre.
    positions = [[CrystalSymmetry.cartesian_operation(image) *
                  SVector{3, T}(view(data[image.conformer_index].mol.atom_coordinates, atom, :))
                  for atom in axes(data[image.conformer_index].mol.atom_coordinates, 1)]
                 for image in basis.images]
    d_min, d_max = T(Inf), zero(T)
    neighbour_sum = zeros(T, length(basis.images))
    for i in eachindex(basis.images), j in i:length(basis.images), cell in eachindex(cells.Delta_R)
        # Exclude only the same physical molecule. Its non-zero periodic copies remain neighbours.
        i == j && all(iszero, cells.indices[cell]) && continue
        separation = basis.images[i].tau - basis.images[j].tau - cells.Delta_R[cell]
        distance = norm(separation)
        # The supplied cells are padded to include all images; only pairs inside R_max enter SR.
        distance <= parameters.R_max || continue
        distance > eps(T) || error("Two molecular images coincide in the short-range sum.")
        d_max = max(d_max, distance)
        # Each off-diagonal image pair contributes to the neighbour sum about both images.
        neighbour_sum[i] += inv(distance)
        i != j && (neighbour_sum[j] += inv(distance))
        # Measure the atomic separation, rather than using the molecular-centre distance for d_min.
        for first in positions[i], second in positions[j]
            d_min = min(d_min, norm(separation + first - second))
        end
    end
    d_min > zero(T) || error("Two nuclei coincide in the short-range neighbour list.")
    return (; d_min, d_max, P = maximum(neighbour_sum),
            r_N = maximum(item.envelope.r_N for item in data),
            sigma_max = maximum(maximum(item.envelope.sigma) for item in data),
            Z = maximum(item.envelope.Z for item in data),
            r_Z = minimum(item.envelope.r_Z for item in data))
end

function short_range_tail(q, data, geometry)
    """
    Estimate the 𝒟 and 𝒥 integrals remaining beyond q = Q using the Gaussian FF envelopes:

        ε_D(Q) ≈ (2 α_EM / π) P/(r_Z² d_min) B_Δ(Q) [B_n(Q) + Z] / Q³,
        ε_J(Q) ≈ (2 α_EM / π) P/(r_Z² d_min) B_f(Q)² / Q³.

    The three 1/q factors represent the two molecular Bessels and the intermolecular Bessel. 
    Integration by parts gives a further inverse phase distance, approximated by 1/d_min. 
    The molecular radii are approximated by r_Z and the remaining angular
    factors by O(1) values, so this is an empirical estimate of the radial error.

    Sum the conformer envelopes and replace each Gaussian-polynomial term divided by q^p
    by its largest value beyond Q. In the units used by the code, that value occurs at

        q_* = max(Q, √max(n-p, 0)/σ).

    Use p = 3 for the nuclear term and p = 3/2 for each electronic factor. If r_Z = 0, retain
    only the intermolecular 1/q decay: use p = 1 for the nuclear term and p = 1/2 for each
    electronic factor, omitting 1/r_Z² from the prefactor.

    # Arguments:
    - q::T: The lower momentum, in keV.
    - data: The prepared molecular quantities and envelopes.
    - geometry::NamedTuple: The molecular and crystal length scales.

    # Returns:
    - Tuple: (ε_D, ε_J), the estimated absolute tail errors in eV.
    """

    T = typeof(q)
    H = T(CrystalLattice.HBAR_C_EV_ANGSTROM / 1000) # Conversion factor from keV to Angstroms.
    power = geometry.r_Z > zero(T) ? T(3) : one(T)
    # half contains the B_F/q^(p/2) envelopes for electronic products, whole contains B_F/q^p.
    half, whole = zeros(T, 3), zeros(T, 3)
    for item in data
        envelope = item.envelope
        for term in eachindex(envelope.sigma)
            sigma, n = envelope.sigma[term], envelope.n[term]
            for (p, values) in ((power / 2, half), (power, whole))
                # Use the largest value beyond q, so a later Gaussian maximum cannot be missed.
                peak = H * sqrt(max(T(n) - p, zero(T))) / sigma
                momentum = max(q, peak)
                radial = (momentum / H)^n * exp(-(sigma * momentum / H)^2 / 2) / momentum^p
                for family in 1:3
                    values[family] += envelope.A[family, term] * radial
                end
            end
        end
    end
    amplitude = T(2 * CrystalLattice.ALPHA_EM * 1000 / π) * H^2 * geometry.P / geometry.d_min
    geometry.r_Z > zero(T) && (amplitude *= H^2 / geometry.r_Z^2)
    # 𝒟 contains both Δ × Z and Δ × n. 𝒥 contains the product of two transition FFs.
    return amplitude * (geometry.Z * whole[3] + half[3] * half[2]), amplitude * half[1]^2
end

function reuse_short_range_grid(saved_q::Vector{T}, spacing::T, reach::T) where {T<:AbstractFloat}
    """
    Choose a uniform SR grid that reuses the most available FF samples among candidate spacings:

        q_k = k h,    k = 0, ..., ceil(Q/h),    h ≤ spacing,    Q = reach.

    Try the largest allowed spacing and spacings commensurate with the saved grid.

    # Arguments:
    - saved_q::Vector{T}: The saved uniform grid starting at zero, in keV.
    - spacing::T: The largest allowed spacing, in keV.
    - reach::T: The required extent, in keV.

    # Returns:
    - Tuple: (q, saved_indices), with q in keV. A non-zero saved_indices[k] gives the saved
      sample for q[k]; zero means that the FF must be evaluated at this node.
    """

    # Check the uniform saved grid, allowing for roundoff in its stored coordinates.
    length(saved_q) >= 2 && iszero(saved_q[1]) || error("The saved q grid must start at zero and contain two points.")
    saved_step = saved_q[2] - saved_q[1]
    tolerance = 8 * eps(T) * max(one(T), saved_q[end])
    saved_step > zero(T) && all(abs.(diff(saved_q) .- saved_step) .<= tolerance) ||
        error("The saved q grid must be uniform and increasing.")
    # Return the same grid if it suffices, so the crystal job can keep the SR integral it already computed.
    if saved_step <= spacing && saved_q[end] >= reach
        return saved_q, collect(eachindex(saved_q))
    end
    # Commensurate spacings reuse saved nodes when extending or refining the grid.
    candidates = T[spacing]
    if saved_step <= spacing
        append!(candidates, saved_step .* (1:floor(Int, spacing / saved_step)))
    else
        push!(candidates, saved_step / ceil(Int, saved_step / spacing))
    end
    best_q, best_indices = T[], Int[]
    best_missing = typemax(Int)
    for step in candidates
        # Round the endpoint upwards so the grid reaches at least the requested qmax.
        q = collect(0:max(1, ceil(Int, reach / step))) .* step
        indices = zeros(Int, length(q))
        missing = 0
        for index in eachindex(q)
            # Reuse a sample only if the nearest saved coordinate agrees to roundoff.
            saved = round(Int, q[index] / saved_step) + 1
            if saved in eachindex(saved_q) && abs(saved_q[saved] - q[index]) <= tolerance
                indices[index] = saved
            else
                missing += 1
            end
        end
        # New FF evaluations are the main cost, so use the shorter grid when that cost is the same.
        if (missing, length(q)) < (best_missing, length(best_q))
            best_q, best_indices, best_missing = q, indices, missing
        end
    end
    return best_q, best_indices
end

function choose_short_range_grid(saved_q::Vector{T}, data, basis, cells, parameters, D_scale, J_scale) where {T<:AbstractFloat}
    """
    Choose a common SR grid for f_λ, Δ and Ξ from the Gaussian envelopes and crystal distances.

    In natural units, the spacing estimate is

        Δq = 2π / [d_max + 2 r_N + σ_alias √(2 ln(1/ε_tot))],
        σ_alias² = 2 σ_max² + 1/(2η²).

    Here ε_tot = parameters.epsilon_tot, d_max + 2 r_N bounds the phase distances, r_N is the largest
    nuclear radius, and σ_alias adds a Gaussian margin for the electronic FFs and the Ewald factor.
    Choose qmax so that both estimated tails satisfy

        ε_D(qmax) ≤ ε_tot D_scale/2,    ε_J(qmax) ≤ ε_tot J_scale/2.

    The scales come from the available grid calculation. The spacing and tail models are
    empirical radial-error estimates at fixed angular cutoff and neighbour list.

    # Arguments:
    - saved_q::Vector{T}: The available FF grid, in keV.
    - data: The prepared molecular quantities and Gaussian envelopes.
    - basis, cells: The excitation basis and real-space neighbour cells.
    - parameters: The Ewald parameters, including the requested relative error.
    - D_scale, J_scale: The cached D and J matrix-element scales, in eV.

    # Returns:
    - Tuple: (q, saved_indices), the SR grid in keV and matching saved indices, with zero for missing nodes.
    """

    # Use the same retained neighbours as the SR integral to measure the phase distances.
    geometry = short_range_geometry(data, basis, cells, parameters)
    iszero(geometry.P) && return saved_q, collect(eachindex(saved_q))
    epsilon = parameters.epsilon_tot
    H = T(CrystalLattice.HBAR_C_EV_ANGSTROM / 1000)
    # Add the two FF Gaussian widths and the width from exp(-q²/(4η²)).
    sigma_alias = sqrt(2 * geometry.sigma_max^2 + inv(2 * parameters.eta^2))
    spacing = 2 * T(π) * H / (geometry.d_max + 2 * geometry.r_N + sigma_alias * sqrt(2 * log(inv(epsilon))))
    lower, upper = zero(T), H / geometry.sigma_max
    # Vanishing density envelopes need no radial extension, even if their reference scales are zero.
    all(iszero, short_range_tail(upper, data, geometry)) &&
        return saved_q, collect(eachindex(saved_q))
    sufficient(q) = begin
        D_tail, J_tail = short_range_tail(q, data, geometry)
        D_tail <= epsilon * D_scale / 2 && J_tail <= epsilon * J_scale / 2
    end
    # First bracket a sufficient qmax by doubling; the Gaussian envelopes eventually decay to zero.
    while !sufficient(upper)
        lower, upper = upper, 2 * upper
        isfinite(upper) || error("The short-range tail estimate did not reach the requested tolerance.")
    end
    # Bisect the decreasing tail estimate, then round onto the uniform grid chosen for sample reuse.
    while upper - lower > 8 * eps(T) * upper
        middle = (lower + upper) / 2
        if sufficient(middle)
            upper = middle
        else
            lower = middle
        end
    end
    return reuse_short_range_grid(saved_q, spacing, upper)
end

function short_range_scales(couplings, D, basis, lattice, long_range)
    """
    Estimate the reference energies used to turn the requested relative radial error into eV:

        D_scale = max_{A_i,s,t} |𝒟_{A_i,st}|,
        J_scale = max_{k,λ_i,λ_j'} |H_{λ_iλ_j'}(k) - δ_{λ_iλ_j'} ΔE_λ - 𝒟_{λ_iλ_j'}|.

    H includes the SR and LR parts of 𝒥, with the self term already subtracted from SR.
    Sample k at fractional reciprocal coordinates (0,0,0), (1/4,0,0), (0,1/4,0) and (1/8,1/8,1/8).

    # Arguments:
    - couplings: The cached SR J couplings, with the self term subtracted.
    - D: The cached total D blocks, in eV.
    - basis, lattice: The crystal excitation basis and lattice.
    - long_range: The Ewald long-range data.

    # Returns:
    - Tuple: (D_scale, J_scale), the largest 𝒟 element and sampled 𝒥 element, in eV.
    """

    T = eltype(basis.Delta_E_monomer)
    # Only the Hamiltonian is needed to read its matrix elements, so do not allocate eigenvectors.
    eigensystem = BlochHamiltonian.BlochEigensystem(basis, long_range; n_neighbour_cells = length(couplings.Delta_R), vecs = false)
    J_scale = zero(T)
    for fraction in (SVector{3, T}(0, 0, 0), SVector{3, T}(1 / 4, 0, 0),
                     SVector{3, T}(0, 1 / 4, 0), SVector{3, T}(1 / 8, 1 / 8, 1 / 8))
        # Convert fractional reciprocal coordinates to the Cartesian wavevector in inverse Å.
        k = SVector{3, T}(transpose(lattice.reciprocal) * fraction)
        BlochHamiltonian.build_bloch_hamiltonian!(eigensystem, basis, k, couplings, long_range)
        for j in axes(eigensystem.H, 2), i in axes(eigensystem.H, 1)
            # Remove the same-molecule 𝒟 and monomer energies to leave the total 𝒥(k).
            value = eigensystem.H[i, j] - basis.D_couplings[i, j]
            i == j && (value -= basis.Delta_E_monomer[i])
            J_scale = max(J_scale, abs(value))
        end
    end
    return maximum(abs, D), J_scale
end

function short_range_coefficients(data, q, saved_indices, R, Delta, Xi, l_max)
    """
    Assemble the transition, difference density and charge FF coefficients on the chosen SR grid:

        R^SR_{ℓm}(q_k) = R^saved_{ℓm}(q_j),    if saved_indices[k] = j > 0,

    and evaluate R^SR_{ℓm}(q_k) from W otherwise. The same rule applies to Δ and Ξ = n - 𝒵, with
    𝒵(q) = ∑_I Z_I exp(i q⋅R_I) the nuclear FF.
    All missing nodes and density families share one R tensor calculation, so each expensive
    Cartesian pair S_{ℓm,ij}(q) is computed once and contracted with all density matrices.
    The retained W tensor and pair geometry are reused, and saved samples are copied unchanged.

    # Arguments:
    - data: The retained W tensor, pair geometry, density matrices, Gaunt path and CPU/GPU settings.
    - q::Vector{T}: The SR grid, in keV.
    - saved_indices::Vector{Int}: The matching saved indices, or zero for missing samples.
    - R, Delta, Xi: The saved transition, Δ and combined Ξ coefficients, indexed by density, q and (ℓ,m).
    - l_max::Int: The largest angular mode.

    # Returns:
    - Tuple: (R_SR, Delta_SR, Xi_SR), with dimensions (n_densities, length(q), (l_max + 1)^2)
      for each family. The angular key is ℓ² + (ℓ + m) + 1.
    """

    # Keep missing nodes in q order, so they can be merged with the saved samples afterwards.
    missing = findall(iszero, saved_indices)
    extra = nothing
    if !isempty(missing)
        momenta = q[missing]
        # Use the same CPU or GPU R calculation, W threshold and density matrices as the saved FFs.
        if data.use_gpu
            W = data.W_ij
            positive, negative = ConstructRTensorGPU.construct_R_tensor_gpu(
                W, data.sigma_ij, data.R_ij_mod, data.R_ij_hat, momenta, l_max,
                maximum(W.lambda), W.n_max, data.gaunt_path, data.density_matrices,
                data.mol.cartesian_term_to_orbital; threshold = data.threshold)
            # Restore the full m range on the CPU before freeing the two GPU tensors.
            all_R = combine_gpu_R_tensor(positive, negative, l_max)
            CUDA.unsafe_free!(positive)
            CUDA.unsafe_free!(negative)
        else
            all_R = construct_R_tensor(data.W_ij, data.sigma_ij, data.R_ij_mod, data.R_ij_hat,
                momenta, l_max, data.gaunt_path, data.density_matrices, data.mol.cartesian_term_to_orbital)
        end
        # Construct the nuclear FF at the new nodes and subtract it from n to form Ξ.
        extra = split_density_R_tensors(all_R, size(R, 1), data.mol.atom_coordinates,
            data.mol.nuclear_charges, momenta, l_max; threshold = data.threshold)
    end
    # Allocate the three output tensors on the common SR grid.
    results = map((R, Delta, Xi)) do saved
        Array{eltype(saved)}(undef, size(saved, 1), length(q), size(saved, 3))
    end
    # Non-zero indices read from the saved grid; zeros read from the next newly evaluated sample.
    extra_index = 0
    for index in eachindex(q)
        saved = saved_indices[index]
        iszero(saved) && (extra_index += 1)
        for family in 1:3
            source = iszero(saved) ? extra[family] : (R, Delta, Xi)[family]
            source_index = iszero(saved) ? extra_index : saved
            @views results[family][:, index, :] .= source[:, source_index, :]
        end
    end
    return results
end
