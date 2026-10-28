# This script computes spherical form factors and the crystal response for a resolved job.

using HDF5
using CUDA
using JSON
using Quaternionic

# Import everything that we need from SCarFFF.
using StaticArrays
using SCarFFF: compute_cartesian_form_factor, compute_fft_form_factor, compute_spherical_form_factor
using SCarFFF.SphericalFormFactor: CrystalImage, build_excitation_basis, build_crystal_lattice,
                                  compute_spherical_form_factor_with_densities,
                                  rotate_R_tensors, rotate_crystal_R_tensors,
                                  compute_coherent_unit_cell_f_sq_lm,
                                  compute_incoherent_unit_cell_f_sq_lm, project_f_sq_lm,
                                  band_energy_range, build_energy_grid, describe_energy_grid,
                                  incoherent_intensity_lm_E, integrate_energy, trim_energy_axis,
                                  apply_structure_function_prefactor!,
                                  enumerate_neighbour_cells, compute_couplings,
                                  compute_crystal_corrections, default_angular_grid,
                                  choose_ewald_parameters, build_ewald_long_range,
                                  D_TRUNCATION, J_TRUNCATION,
                                  compute_ewald_D, subtract_self_term!,
                                  check_dipole_against_oscillator_strength,
                                  image_translation_span, supercell_radius,
                                  set_D_couplings!,
                                  choose_short_range_grid, short_range_scales, short_range_coefficients,
                                  derive_symmetry_operations, build_stars,
                                  star_reduction_factor, choose_compatible_phi_count
using SCarFFF.SphericalFormFactor.PrecomputeGaunt: precompute_gaunt_coefficients
using SCarFFF.SphericalFormFactor.CrystalLattice: KEV_TO_INV_ANGSTROM
using SCarFFF.SphericalFormFactor.BlochHamiltonian: BlochEigensystem, solve_bloch_energies!
using SCarFFF.StageTimings: print_stage_timings



function manifold_spreads(uncoupled::Vector{T}, coupled::Vector{T}) where {T<:AbstractFloat}
    """
    The width of each Davydov multiplet.

    A monomer transition appears once per molecule in the cell, so with no couplings the spectrum is
    a set of degenerate blocks, one block per transition s holding n_images copies of ΔE_s. Turning
    the couplings on splits each block into a Davydov multiplet. For the block of crystal states
    n = a to b that came from one monomer energy, this returns its width

        W_s = max_{n = a..b} ΔE_n - min_{n = a..b} ΔE_n.

    D moves the mean of a block, J sets its width.

    # Arguments:
    - uncoupled::Vector{T}: The uncoupled energies, ascending.
    - coupled::Vector{T}: The coupled energies, ascending.

    # Returns:
    - Vector{T}: One width per block, in eV.
    """

    # Walk through the uncoupled energies, closing a block each time the value changes.
    spreads = T[]
    start = 1
    for state in 2:(length(uncoupled) + 1)
        if state > length(uncoupled) || !isapprox(uncoupled[state], uncoupled[start]; rtol = 1e-8)
            block = @view coupled[start:(state - 1)]
            push!(spreads, maximum(block) - minimum(block))
            start = state
        end
    end

    return spreads
end

function print_gamma_point_shifts(
        basis, lattice, couplings, long_range, ::Type{T},
    ) where {T<:AbstractFloat}
    """
    Print the crystal states at Γ beside the monomer energies they came from.

    With no couplings H is diag(E), so its spectrum is just the monomer energies repeated once per
    molecule in the cell. Turning J and D on and solving at k = 0 gives the crystal spectrum. Both
    are sorted ascending, so comparing them state by state shows what the crystal did.

    # Arguments:
    - basis: The localised excitation basis.
    - lattice: The crystal lattice, unused beyond matching the solver signature.
    - couplings: The real-space couplings, or nothing.
    - long_range: The Ewald long-range data, or nothing.
    - T::Type: The floating point type.

    # Returns:
    - Nothing. The table is printed.
    """

    # Get the crystal spectrum at the Gamma point.
    n_lambda_i = length(basis.Delta_E_monomer)
    n_neighbour_cells = couplings === nothing ? 0 : length(couplings.Delta_R)
    eigensystem = BlochEigensystem(basis, long_range; n_neighbour_cells = n_neighbour_cells, vecs = false)
    solve_bloch_energies!(eigensystem, basis, SVector{3, T}(0, 0, 0), couplings, long_range)

    # Get the monomer spectrum at the Gamma point.
    uncoupled = sort(basis.Delta_E_monomer)
    coupled = eigensystem.Delta_E

    # Fixed-width columns, with a rule between manifolds so the Davydov blocks are visible.
    cell(x) = lpad(string(round(x, digits = 4)), 15)
    rule = "  " * "\u2500"^7 * "\u253c" * "\u2500"^16 * "\u253c" * "\u2500"^16 * "\u253c" * "\u2500"^16

    println("\nCrystal states at \u0393 (k = 0)")
    println("  " * rpad("state", 7) * "\u2502" * lpad("no coupling", 16) * "\u2502" *
            lpad("with coupling", 16) * "\u2502" * lpad("\u0394", 16))
    println("  " * rpad("", 7) * "\u2502" * lpad("[eV]", 16) * "\u2502" *
            lpad("[eV]", 16) * "\u2502" * lpad("[eV]", 16))
    println(rule)
    for state in 1:n_lambda_i
        if state > 1 && !isapprox(uncoupled[state], uncoupled[state - 1]; rtol = 1e-8)
            println(rule)
        end
        println("  " * rpad(state, 7) * "\u2502" * cell(uncoupled[state]) * " \u2502" *
                cell(coupled[state]) * " \u2502" * cell(coupled[state] - uncoupled[state]) * " ")
    end
    println(rule)

    spreads = manifold_spreads(uncoupled, coupled)
    println("  largest shift " * string(round(maximum(abs, coupled .- uncoupled), digits = 4)) *
            " eV, largest Davydov splitting " * string(round(maximum(spreads), digits = 4)) * " eV")

    return nothing
end


function ordinal_suffix(n::Int)::String
    """
    Return the ordinal suffix for a given integer, e.g. 1 -> "st", 2 -> "nd", 3 -> "rd", 4 -> "th".

    # Arguments:
    - n::Int: The integer to get the ordinal suffix for.

    # Returns:
    - suffix::String: The ordinal suffix.
    """

    if n % 10 == 1 && n % 100 != 11
        return "st"
    elseif n % 10 == 2 && n % 100 != 12
        return "nd"
    elseif n % 10 == 3 && n % 100 != 13
        return "rd"
    else
        return "th"
    end
end

function build_dominant_group_images(crystal_metadata, conformer_labels::Vector{String}, ::Type{T}) where {T<:AbstractFloat}
    """
    Collect the molecules of the dominant disorder group as CrystalImage objects, for the coherent
    crystal treatment.

    The coherent Frenkel exciton treatment assumes a perfect, ordered crystal, so for now it runs on
    the highest-occupancy disorder group alone. Every group is still parsed and carried in the
    metadata, so that a proper (Monte Carlo) disorder treatment can use them later.

    # Arguments:
    - crystal_metadata::Dict{String, Any}: The crystal metadata read from JSON.
    - conformer_labels::Vector{String}: The conformer labels, whose order sets the conformer indices.
    - T::Type: The floating point type to use.

    # Returns:
    - images::Vector{CrystalImage{T}}: The molecules of the dominant group.
    - group_name::String: The name of the group used.
    - occupancy::T: That group's occupancy.
    """

    haskey(crystal_metadata, "dominant_group") ||
        error("The crystal metadata has no 'dominant_group' entry. Re-run td_dft.py to regenerate crystal_metadata.json with the lattice and translation data the coherent path needs.")

    # Get the dominant group, and turn labels to indices (A -> 1, B -> 2,...).
    dominant_group = string(crystal_metadata["dominant_group"])
    label_to_index = Dict(label => idx for (idx, label) in enumerate(conformer_labels))

    # Find the index of the dominant group.
    group_index = findfirst(g -> string(g["group"]) == dominant_group, crystal_metadata["disorder_groups"])
    group_index === nothing && error("The dominant disorder group '$(dominant_group)' is not present in the metadata.")
    disorder_group = crystal_metadata["disorder_groups"][group_index]

    # Initialise the list of images.
    images = CrystalImage{T}[]
    for molecule in disorder_group["molecules"]
        # Check that the molecule has all the required data.
        haskey(molecule, "translation") ||
            error("A molecule in the crystal metadata has no 'translation'. Re-run td_dft.py to regenerate crystal_metadata.json.")

        label = molecule["conformer_label"]
        haskey(label_to_index, label) || error("No conformer found for label $(label).")

        # Add the data of the molecule to the image list.
        push!(images, CrystalImage{T}(
            label_to_index[label],
            Quaternionic.rotor(T.(molecule["proper_quaternion"])),
            T(molecule["det_rotation"]),
            SVector{3, T}(T.(molecule["translation"])),
        ))
    end

    isempty(images) && error("The dominant disorder group '$(dominant_group)' contains no molecules.")

    return images, dominant_group, T(disorder_group["occupancy"])
end

function run_crystals(job)
    """
    Compute the conformer form factors and crystal response for each selected crystal.

    # Arguments:
    - job::Dict: The resolved crystal job.

    # Returns:
    - Vector{String}: The numbers of crystals whose calculations failed.
    """

    settings, crystal = job["spherical"], job["crystal"]
    use_gpu = job["execution"]["gpu"]
    use_gpu && !CUDA.functional() && error("A GPU was requested but CUDA is not functional.")
    method = "spherical"
    crystal_mode = true
    crystal_order = crystal["order"]
    coupling_method = crystal["coupling_method"]
    coupling_cutoff = crystal["coupling_cutoff"]
    epsilon_tot = crystal["ewald_epsilon_tot"]
    ewald_eta = crystal["ewald_eta"]
    ewald_cost_ratio = crystal["ewald_cost_ratio"]
    epsilon_b = crystal["epsilon_b"] == "inf" ? Inf : crystal["epsilon_b"]
    q_bc = crystal["q_bc"]
    energy_step = crystal["energy_step"]
    need_structure_factor = crystal["structure_function"]
    need_incoherent_reference = crystal["incoherent_reference"]
    band_range_star_stride = crystal["band_range_star_stride"]
    band_range_q_stride = crystal["band_range_q_stride"]
    no_couplings = !crystal["couplings"]
    selected_transitions = job["transitions"] == "all" ? collect(1:job["nstates"]) : Int.(job["transitions"])
    force_recomp = settings["force_recomputation"]
    T = job["execution"]["precision"] == "float32" ? Float32 : Float64

    # Cast variables to correct types.
    typed_zero = zero(T)
    type_suffix = T == Float32 ? "_f32" : "_f64"

    # Check that crystal mode isn't asked for outside of spherical mode.
    if crystal_mode && method != "spherical"
        error("Crystal mode is only supported for the spherical method.")
    end


    entries = [(number=system["number"], kind="crystal", display_name=system["label"],
                input_value=system["input"]) for system in job["systems"]]
    run_directory = job["output"]
    failures = String[]

    # Process each molecule or crystal.
    for entry in entries
        mol_num = entry.number
        mol_output_dir = joinpath(run_directory, string(mol_num))

        println("="^50)
        println("Computing form factor for $(entry.kind) $mol_num of $(length(entries)), with label $(entry.display_name).")
        println("="^50)
        println()

        # Check if the TD-DFT calculation failed for this molecule or crystal.
        failed_marker = joinpath(mol_output_dir, ".tddft_failed")
        if isfile(failed_marker)
            @warn "TD-DFT was marked as failed for entry $mol_num. Skipping."
            push!(failures, string(mol_num))
            continue
        end

        # Check if TD-DFT or crystal data is missing.
        if crystal_mode
            crystal_metadata_path = joinpath(mol_output_dir, "crystal_metadata.json")
            if !isfile(crystal_metadata_path)
                @warn "No crystal metadata found at $(crystal_metadata_path). Skipping entry $mol_num."
                push!(failures, string(mol_num))
                continue
            end
        else
            td_h5 = joinpath(mol_output_dir, "td_dft_results$(type_suffix).h5")
            if !isfile(td_h5)
                @warn "No TD-DFT results found at $(td_h5). Skipping entry $mol_num."
                continue
            end
        end

        # Wrap the form factor computation in a try/catch so that a single molecule failing
        # (e.g. due to missing or corrupt data) does not crash the entire batch.
        try

        # Parse the transition indices.
        transition_indices = Int[]
        transition_index = 1
        transition_suffix = "st"
        if !crystal_mode
            transition_indices = selected_transitions
            transition_index = transition_indices[1]
            transition_suffix = ordinal_suffix(transition_index)
        end

        if method == "spherical"
            # Parse spherical-specific parameters.
            q_max = settings["q_max"]
            N_q = settings["n_q"]
            N_theta = settings["n_theta"]
            N_phi = settings["n_phi"]
            l_max = settings["l_max"]

            # Size the angular grid from l_max unless it was set explicitly.
            default_theta, default_phi = max.(3, default_angular_grid(l_max))
            N_theta = N_theta > 0 ? N_theta : default_theta
            N_phi = N_phi > 0 ? N_phi : default_phi
            threshold_val = T(settings["threshold"])
            need_grid = "f_s" in settings["save"]
            need_R = "R_tensor" in settings["save"]
            if crystal_mode
                # Both crystal orders rotate each conformer's R tensor into every image's
                # orientation, so it has to be kept even if it was not asked for as an output.
                need_R = true
            end

            # Widen the grid by a few points if necessary to maximise the symmetry reduction. Only
            # the coherent order diagonalises, so only it benefits.
            if crystal_mode && crystal_order == "coherent"
                grid_metadata = JSON.parsefile(joinpath(mol_output_dir, "crystal_metadata.json"))
                grid_images, _, _ = build_dominant_group_images(
                    grid_metadata, String.(grid_metadata["conformer_labels"]), T)
                grid_lattice = build_crystal_lattice(
                    reduce(vcat, [reshape(T.(row), 1, 3) for row in grid_metadata["lattice"]]), T)
                widened_N_phi = choose_compatible_phi_count(
                    N_theta, N_phi, derive_symmetry_operations(grid_images, grid_lattice))
                if widened_N_phi != N_phi
                    println("Widening N_phi from $(N_phi) to $(widened_N_phi) so the crystal symmetry " *
                            "operations land on grid nodes, which buys back diagonalisations for " *
                            "$(round(100 * (widened_N_phi / N_phi - 1), digits = 1))% more grid points.")
                    N_phi = widened_N_phi
                end
            end

            # Define the momentum grid.
            q_grid = collect(range(typed_zero, T(q_max), length=N_q))
            theta_grid = collect(range(typed_zero, T(π), length=N_theta))
            phi_grid = collect(range(typed_zero, T(2π), length=N_phi))

            if crystal_mode
                # Find the metadata containing groups and rotations.
                crystal_metadata_path = joinpath(mol_output_dir, "crystal_metadata.json")
                crystal_metadata = JSON.parsefile(crystal_metadata_path)
                conformer_labels = String.(crystal_metadata["conformer_labels"])
                if isempty(conformer_labels)
                    error("No conformers found in $(crystal_metadata_path).")
                end

                first_td_h5 = joinpath(mol_output_dir, "conformers", conformer_labels[1], "td_dft_results$(type_suffix).h5")
                transition_indices = selected_transitions
                transition_index = transition_indices[1]
                transition_suffix = ordinal_suffix(transition_index)

                if length(transition_indices) == 1
                    println("Computing the spherical crystal form factor for the $(transition_index)$(transition_suffix) transition of $(entry.display_name) up to maximum angular mode l = $(l_max).")
                else
                    println("Computing the spherical crystal form factor for $(length(transition_indices)) transitions of $(entry.display_name) up to maximum angular mode l = $(l_max).")
                    println("Transition indices: $(transition_indices)")
                end
                println("Grid: q ∈ [0, $(q_max)] keV with $(N_q) points,")
                println("      θ ∈ [0, π] with $(N_theta) points,")
                println("      φ ∈ [0, 2π] with $(N_phi) points.")

                # Create a results dict for each conformer group.
                conformer_results = Dict{String, Any}()

                # Only the Ewald split has a Q = 0 term, so only it needs the dipole at q_bc.
                need_q_bc = crystal_order == "coherent" && !no_couplings && coupling_method == "ewald"

                # Compute the spherical form factor for each conformer group.
                for conformer_label in conformer_labels
                    td_h5 = joinpath(mol_output_dir, "conformers", conformer_label, "td_dft_results$(type_suffix).h5")
                    if !isfile(td_h5)
                        error("No conformer TD-DFT results found at $(td_h5).")
                    end

                    # The coherent correction also needs Δ and Ξ. They go through the same R
                    # contraction, but remain internal rather than changing the molecular output.
                    need_Delta_and_Xi = crystal_order == "coherent" && !no_couplings
                    Delta_and_Xi = nothing
                    if need_Delta_and_Xi
                        R_tensor, f_s, f_sq_lm, Delta_E_eV, Delta_and_Xi =
                            compute_spherical_form_factor_with_densities(
                                q_grid,
                                theta_grid,
                                phi_grid,
                                l_max,
                                td_h5,
                                transition_indices=transition_indices,
                                force_recomputation=force_recomp,
                                threshold=threshold_val,
                                use_gpu=use_gpu,
                                need_grid=need_grid,
                                need_R=need_R,
                                need_f_sq_lm=true,
                                need_short_range=need_q_bc
                            )
                    else
                        R_tensor, f_s, f_sq_lm, Delta_E_eV = compute_spherical_form_factor(
                            q_grid,
                            theta_grid,
                            phi_grid,
                            l_max,
                            td_h5,
                            transition_indices=transition_indices,
                            force_recomputation=force_recomp,
                            threshold=threshold_val,
                            use_gpu=use_gpu,
                            need_grid=need_grid,
                            need_R=need_R,
                            need_f_sq_lm=true
                        )
                    end

                    # The Ewald Q = 0 term needs the transition dipole, which is the q → 0 limit of f_{1m}(q)/q.
                    R_bc = nothing
                    Delta_bc = nothing
                    Xi_bc = nothing
                    if need_q_bc
                        # We have to include the q = 0 point since there's a check for neutrality at the origin.
                        R_both, _, _, _, Delta_and_Xi_bc =
                            compute_spherical_form_factor_with_densities(
                                T[0, q_bc],
                                theta_grid,
                                phi_grid,
                                l_max,
                                td_h5,
                                transition_indices=transition_indices,
                                force_recomputation=force_recomp,
                                threshold=threshold_val,
                                use_gpu=use_gpu,
                                need_grid=false,
                                need_R=true,
                                need_f_sq_lm=false
                            )
                        R_bc = R_both[:, 2:2, :]
                        Delta_bc = Delta_and_Xi_bc.Delta_R[:, 2:2, :]
                        Xi_bc = Delta_and_Xi_bc.Xi_R[:, 2:2, :]

                        # TD-DFT reports the oscillator strength, which fixes the dipole without
                        # going near the form factor, so it catches a broken ℓ = 1 coefficient.
                        f_osc = h5open(td_h5, "r") do io
                            T.(read(io, "f_osc"))[transition_indices]
                        end
                        dipole_error = check_dipole_against_oscillator_strength(
                            R_bc, T(q_bc * KEV_TO_INV_ANGSTROM),
                            T.(Delta_E_eV), f_osc, l_max)
                        println("Conformer $(conformer_label): the dipole at q_bc = $(q_bc) keV matches " *
                                "the oscillator strength to $(round(dipole_error * 100, digits = 4))%.")
                    end

                    if job["verification"]["oscillator_strength"]
                        n_fit = searchsortedlast(q_grid, job["verification"]["q_max_fit"])
                        n_fit >= 2 || error("The oscillator strength check needs two q points below q_max_fit.")
                        reference = h5open(io -> read(io, "f_osc"), td_h5, "r")
                        for (i, s) in enumerate(transition_indices)
                            mean = perform_angular_average(abs2.(f_s[i, 1:n_fit, :, :]), theta_grid, phi_grid)
                            moment = compute_transition_dipole_moment(mean, q_grid[1:n_fit])
                            fitted = 2 * ELECTRON_MASS_KEV * Delta_E_eV[i] * 0.001 * moment
                            println("Conformer $conformer_label, transition $s: fitted oscillator strength $fitted; TD-DFT $(reference[s]).")
                        end
                    end

                    conformer_results[conformer_label] = (
                        R_tensor = R_tensor,
                        f_s = f_s,
                        f_sq_lm = f_sq_lm,
                        Delta_E_eV = Delta_E_eV,
                        Delta_and_Xi = Delta_and_Xi,
                        R_bc = R_bc,
                        Delta_bc = Delta_bc,
                        Xi_bc = Xi_bc,
                    )

                    # Write the results to HDF5 for this conformer.
                    for (batch_idx, transition_idx) in enumerate(transition_indices)
                        transition_output_dir = joinpath(mol_output_dir, "conformers", conformer_label, string(transition_idx))
                        mkpath(transition_output_dir)

                        transition_energy = Delta_E_eV[batch_idx]
                        output_path = joinpath(transition_output_dir, "fs_grid$(type_suffix).h5")
                        h5open(output_path, "w") do io
                            if f_s !== nothing
                                f_s_slice = f_s[batch_idx, :, :, :]
                                write(io, "f_s", f_s_slice)
                                write_quantity(io, "theta_grid", theta_grid, "rad") # Radians.
                                write_quantity(io, "phi_grid", phi_grid, "rad") # Radians.
                            end
                            if R_tensor !== nothing
                                R_slice = R_tensor[batch_idx, :, :]
                                write(io, "R_tensor", R_slice)
                            end
                            if f_sq_lm !== nothing
                                f_sq_lm_slice = f_sq_lm[batch_idx, :, :]
                                write(io, "f_sq_lm", f_sq_lm_slice)
                            end
                            write_quantity(io, "q_grid", q_grid, "keV") # keV.
                            write(io, "transition_index", transition_idx)
                            write_quantity(io, "Delta_E_eV", transition_energy, "eV")
                        end
                    end
                end

                #Run the crystal computation. Coherent solves the Frenkel exciton Bloch problem and mixes them before
                # squaring, incoherent adds them in intensity with no phases and no mixing.
                crystal_results = nothing
                begin
                    haskey(crystal_metadata, "lattice") ||
                        error("The crystal metadata has no 'lattice' entry, which the crystal path needs. Re-run td_dft.py to regenerate crystal_metadata.json.")

                    # Construct the lattice and reciprocal lattice.
                    lattice_matrix = reduce(vcat, [reshape(T.(row), 1, 3) for row in crystal_metadata["lattice"]])
                    lattice = build_crystal_lattice(lattice_matrix, T)

                    # Build the list of images of the dominant group.
                    images, dominant_group, dominant_occupancy =
                        build_dominant_group_images(crystal_metadata, conformer_labels, T)

                    if crystal_order == "coherent"
                        println("Solving the Frenkel exciton Bloch problem for disorder group $(dominant_group) ($(length(images)) molecules, occupancy $(dominant_occupancy)).")
                    else
                        println("Adding the images of disorder group $(dominant_group) in intensity ($(length(images)) molecules, occupancy $(dominant_occupancy)).")
                    end

                    # Get the R tensors for each conformer.
                    conformer_R_tensors = [conformer_results[label].R_tensor for label in conformer_labels]
                    any(isnothing, conformer_R_tensors) &&
                        error("The crystal path needs the R tensor for every conformer, but at least one is missing.")

                    conformer_energy_lists = [T.(conformer_results[label].Delta_E_eV) for label in conformer_labels]
                    basis = build_excitation_basis(images, conformer_energy_lists)

                    # Rotate each conformer's coefficients into every image's orientation using D matrices.
                    # This keeps all images on the same unrotated q grid.
                    stage_times = Pair{String, Float64}[]
                    rotated_Delta = nothing
                    rotated_Xi = nothing
                    rotated_R_bc = nothing
                    rotated_Delta_bc = nothing
                    rotated_Xi_bc = nothing
                    if crystal_order == "coherent" && !no_couplings
                        conformer_Delta = [conformer_results[label].Delta_and_Xi.Delta_R
                                                for label in conformer_labels]
                        conformer_Xi = [conformer_results[label].Delta_and_Xi.Xi_R
                                        for label in conformer_labels]
                        rotate_timing = @timed rotate_crystal_R_tensors(
                            conformer_R_tensors, conformer_Delta, conformer_Xi,
                            images, basis, l_max)
                        rotated_R, rotated_Delta, rotated_Xi = rotate_timing.value
                    else
                        rotate_timing = @timed rotate_R_tensors(
                            conformer_R_tensors, images, basis, l_max)
                        rotated_R = rotate_timing.value
                    end
                    push!(stage_times, "rotate R tensors" => rotate_timing.time)

                    # Rotate the dipole terms.
                    if need_q_bc
                        conformer_R_bc = [conformer_results[label].R_bc for label in conformer_labels]
                        conformer_Delta_bc = [conformer_results[label].Delta_bc
                                                   for label in conformer_labels]
                        conformer_Xi_bc = [conformer_results[label].Xi_bc for label in conformer_labels]
                        rotated_R_bc, rotated_Delta_bc, rotated_Xi_bc = rotate_crystal_R_tensors(
                            conformer_R_bc, conformer_Delta_bc, conformer_Xi_bc,
                            images, basis, l_max)
                    end

                    # The Brillouin zone folding works in inverse Angstroms, matching the lattice.
                    q_grid_invA = T(KEV_TO_INV_ANGSTROM) .* q_grid
                    q_bc_invA = T(q_bc * KEV_TO_INV_ANGSTROM)

                if crystal_order != "coherent"
                    # Nothing here depends on k, so there is no Bloch problem to solve: no couplings,
                    # no Ewald sum, no symmetry stars and no diagonalisation.
                    incoherent_timing = @timed compute_incoherent_unit_cell_f_sq_lm(
                        rotated_R, basis, q_grid_invA, theta_grid, phi_grid, l_max;
                        need_grid = need_grid)
                    crystal_state_f_sq_lm, I_uc = incoherent_timing.value
                    push!(stage_times, "incoherent f_sq_lm" => incoherent_timing.time)

                    # Build the energy axis and spread the f_sq_lm coefficients over it.
                    # We use the raw monomer energies here for the energy grid, since there are no bands.
                    crystal_S_lm = nothing
                    crystal_energy_grid = nothing
                    if need_structure_factor
                        # basis.Delta_E_monomer is indexed by λ_i, and the first n_trans entries are just the energies for one molecule.
                        transition_energies = basis.Delta_E_monomer[1:basis.n_transitions]
                        crystal_energy_grid = build_energy_grid(
                            minimum(transition_energies), maximum(transition_energies),
                            T(energy_step))
                        structure_timing = @timed incoherent_intensity_lm_E(
                            crystal_state_f_sq_lm, transition_energies, crystal_energy_grid)
                        crystal_S_lm = structure_timing.value
                        # The 2π/V_cell that makes this the structure function rather than the bare sum.
                        apply_structure_function_prefactor!(crystal_S_lm, lattice.V_cell)
                        push!(stage_times, "structure function f_sq_lm(q, E)" => structure_timing.time)
                    end

                    print_stage_timings("Crystal stage timings", stage_times)
                    crystal_energy_grid === nothing ||
                        println("\n" * describe_energy_grid(crystal_energy_grid))

                    crystal_results = (
                        order = "incoherent",
                        basis = basis,
                        lattice = lattice,
                        dominant_group = dominant_group,
                        dominant_occupancy = dominant_occupancy,
                        state_f_sq_lm = crystal_state_f_sq_lm,
                        # An unmixed crystal already is the incoherent reference, so saving a copy
                        # of it under a second name would tell a reader nothing.
                        incoherent_f_sq_lm = nothing,
                        f_uc = nothing,
                        I_uc = I_uc,
                        S_lm = crystal_S_lm,
                        energy_grid = crystal_energy_grid,
                        band_summary = nothing,
                        couplings = nothing,
                        cells = nothing,
                        q_short_range = nothing,
                        ewald_parameters = nothing,
                        long_range = nothing,
                    )
                else

                    # Build the intermolecular couplings, unless they have been switched off.
                    crystal_couplings = nothing
                    J_parameters = nothing
                    D_parameters = nothing
                    crystal_long_range = nothing
                    cells = nothing
                    q_sr = nothing
                    if no_couplings
                        println("Intermolecular corrections J and D are switched off, so the bands will be flat at the monomer energies.")
                    else
                        if coupling_method == "ewald"
                            # J and D converge at different rates, so each gets its own cutoffs to save wasted computation.
                            J_parameters = choose_ewald_parameters(
                                lattice, length(basis.images), epsilon_tot, J_TRUNCATION;
                                eta = ewald_eta, cost_ratio = ewald_cost_ratio)
                            D_parameters = choose_ewald_parameters(
                                lattice, length(basis.images), epsilon_tot, D_TRUNCATION;
                                eta = ewald_eta, cost_ratio = ewald_cost_ratio)
                            # Pad the cutoff to not miss any molecules. Extra molecules will be dropped later.
                            real_cutoff = max(J_parameters.R_max, D_parameters.R_max) +
                                          image_translation_span(basis)
                        else
                            real_cutoff = T(coupling_cutoff)
                        end

                        cells = enumerate_neighbour_cells(lattice, real_cutoff)

                        # Precompute the Gaunt coefficients for the coupling.
                        coupling_gaunt_path = joinpath(@__DIR__, "..", "data", "gaunt_coefficients",
                                                       "Gaunt_l1$(l_max)_l2$(2 * l_max)_l3$(l_max)$(type_suffix).h5")
                        if force_recomp || !isfile(coupling_gaunt_path)
                            precompute_gaunt_coefficients(l_max, 2 * l_max, l_max, coupling_gaunt_path, T)
                        end

                        if J_parameters === nothing
                            println("Computing intermolecular corrections directly out to $(coupling_cutoff) Å ($(length(cells.Delta_R)) cells).")
                            correction_timing = @timed compute_crystal_corrections(
                                rotated_R, rotated_Delta, rotated_Xi,
                                basis, cells, q_grid_invA, l_max, coupling_gaunt_path)
                            crystal_couplings, D_blocks = correction_timing.value
                            set_D_couplings!(basis, D_blocks)
                            push!(stage_times, "couplings J and D" => correction_timing.time)
                        else
                            tau_of_lambda_i = [basis.images[basis.image_of[lambda_i]].tau
                                             for lambda_i in 1:length(basis.Delta_E_monomer)]
                            long_range_timing = @timed build_ewald_long_range(
                                rotated_R, rotated_R_bc, tau_of_lambda_i, basis.image_of, lattice,
                                q_grid_invA, q_bc_invA, l_max,
                                J_parameters; epsilon_b = epsilon_b)
                            crystal_long_range = long_range_timing.value
                            push!(stage_times, "Ewald long-range setup" => long_range_timing.time)

                            # Print a summary of both Ewald splits, against the 3x3x3 supercell as a
                            # familiar length to read the real-space cutoffs against.
                            println("Ewald split at ε_tot = $(epsilon_tot), one per coupling " *
                                    "($(length(cells.Delta_R)) cells searched, padded by the cell span, " *
                                    "3x3x3 supercell radius $(round(supercell_radius(lattice), digits = 2)) Å):")
                            for (name, split) in (("J", J_parameters), ("D", D_parameters))
                                expected_G = lattice.V_cell * split.G_max^3 / (6 * π^2)
                                println("  $(name): η = $(round(split.eta, digits = 4)) Å^-1, " *
                                        "R_max = $(round(split.R_max, digits = 2)) Å, " *
                                        "G_max = $(round(split.G_max, digits = 4)) Å^-1 " *
                                        "(N_G ≈ $(round(Int, expected_G))).")
                            end
                            println("  J long-range table: $(length(crystal_long_range.G_vectors)) " *
                                    "candidate G, ℓ ceiling $(crystal_long_range.l_max_lr) of $(l_max).")

                            correction_timing = @timed compute_crystal_corrections(
                                rotated_R, rotated_Delta, rotated_Xi,
                                basis, cells, q_grid_invA, l_max, coupling_gaunt_path;
                                J_parameters = J_parameters, D_parameters = D_parameters)
                            crystal_couplings, D_blocks_short_range = correction_timing.value
                            push!(stage_times, "short-range J and D(ΔR)" => correction_timing.time)
                            subtract_self_term!(crystal_couplings, cells, crystal_long_range.J_self)

                            D_long_range_timing = @timed compute_ewald_D(
                                rotated_Delta, rotated_Xi,
                                rotated_Delta_bc, rotated_Xi_bc,
                                q_grid_invA, q_bc_invA, basis.n_transitions, l_max,
                                crystal_long_range, D_parameters)
                            # Use the available grid D and total J(k) to convert the relative SR tolerance to eV.
                            D_scale, J_scale = short_range_scales(
                                crystal_couplings, D_blocks_short_range .+ D_long_range_timing.value,
                                basis, lattice, crystal_long_range)
                            sr_data = [conformer_results[label].Delta_and_Xi.short_range_data
                                       for label in conformer_labels]
                            # Choose one SR grid for all FF families, marking nodes already present in q_grid.
                            q_sr, saved_indices = choose_short_range_grid(
                                q_grid, sr_data, basis, cells, J_parameters, D_parameters,
                                D_scale, J_scale)
                            missing_points = count(iszero, saved_indices)
                            println("SR radial grid: $(length(q_sr)) points through $(round(q_sr[end], digits = 2)) keV, " *
                                    "Δq = $(round(q_sr[2] - q_sr[1], digits = 5)) keV, " *
                                    "$(missing_points) new FF samples per conformer (estimated ε = $(epsilon_tot)).")
                            # If the original grid suffices, keep the SR J and D already computed above.
                            if q_sr != q_grid
                                sr_timing = @timed begin
                                    # Evaluate only missing q nodes, sharing S_ij between f_s, ΔN and Ξ.
                                    coefficients = [short_range_coefficients(
                                        sr_data[index], q_sr, saved_indices, conformer_R_tensors[index],
                                        conformer_Delta[index], conformer_Xi[index], l_max)
                                        for index in eachindex(conformer_labels)]
                                    # Rotate the new molecular coefficients into the same crystal images.
                                    sr_R, sr_Delta, sr_Xi = rotate_crystal_R_tensors(
                                        [item[1] for item in coefficients], [item[2] for item in coefficients],
                                        [item[3] for item in coefficients], images, basis, l_max)
                                    compute_crystal_corrections(sr_R, sr_Delta, sr_Xi, basis, cells,
                                        T(KEV_TO_INV_ANGSTROM) .* q_sr, l_max, coupling_gaunt_path;
                                        J_parameters = J_parameters, D_parameters = D_parameters)
                                end
                                crystal_couplings, D_blocks_short_range = sr_timing.value
                                # The refined couplings replace the earlier SR result, so subtract their self term once.
                                subtract_self_term!(crystal_couplings, cells, crystal_long_range.J_self)
                                push!(stage_times, "SR grid and J/D refinement" => sr_timing.time)
                            end
                            # Add the LR D from the original FF table to the final SR D blocks.
                            set_D_couplings!(
                                basis, D_blocks_short_range .+ D_long_range_timing.value)
                            push!(stage_times, "Ewald long-range D" => D_long_range_timing.time)
                        end
                    end

                    # Print a summary of the timing, and symmetry statistics.
                    symmetry_timing = @timed begin
                        operations = derive_symmetry_operations(basis.images, lattice)
                        build_stars(theta_grid, phi_grid, operations)
                    end
                    stars = symmetry_timing.value
                    push!(stage_times, "symmetry stars" => symmetry_timing.time)
                    println("Symmetry: $(length(stars.operations)) usable operations, " *
                            "$(length(stars.irreducible_directions)) stars over " *
                            "$(length(theta_grid) * length(phi_grid)) directions, " *
                            "$(round(star_reduction_factor(stars, length(theta_grid), length(phi_grid)), digits = 2))x " *
                            "fewer diagonalisations.")

                    # Print a warning about even ϕ grids losing symmetry speedup.
                    iseven(length(phi_grid)) && println(
                        "  Note: N_phi = $(length(phi_grid)) is even, so ϕ -> ϕ + π falls between grid " *
                        "points and roughly half the symmetry operations are unusable. An odd N_phi " *
                        "(the default is 4 * l_max + 1) would recover them at no cost in accuracy.")

                    # Build the energy grid by sampling the bands. We don't do all k points or stars as this is
                    # too expensive, but instead stride over them, and then buffer a bit. In practice, this is
                    # usually exact, or close to. The extra points cost no computation or memory at the end, as they
                    # are skipped in calculation, and trimmed.
                    crystal_energy_grid = nothing
                    if need_structure_factor
                        range_timing = @timed band_energy_range(
                            basis, lattice, q_grid_invA, stars, crystal_couplings,
                            crystal_long_range; star_stride = band_range_star_stride,
                            q_stride = band_range_q_stride)
                        energy_low, energy_high, energy_margin = range_timing.value
                        crystal_energy_grid = build_energy_grid(
                            energy_low - energy_margin, energy_high + energy_margin, T(energy_step))
                        push!(stage_times, "band energy range" => range_timing.time)
                        println("\nSampled band range $(round(energy_low, digits = 4)) to " *
                                "$(round(energy_high, digits = 4)) eV from every " *
                                "$(band_range_star_stride) star and every " *
                                "$(band_range_q_stride) q, widened by " *
                                "$(round(1000 * energy_margin, digits = 1)) meV each side.")
                    end

                    # This streams over q internally, projecting each block onto real spherical
                    # spherical harmonics as it goes, so the full |f|^2 grid is never held in memory. Overwritten
                    # need grid is specfied.
                    coherent_timing = @timed compute_coherent_unit_cell_f_sq_lm(
                        rotated_R, basis, lattice, q_grid_invA, theta_grid, phi_grid, l_max,
                        stars;
                        need_grid = need_grid, couplings = crystal_couplings,
                        long_range = crystal_long_range, energy_grid = crystal_energy_grid)
                    crystal_state_f_sq_lm, crystal_band_summary, f_uc, crystal_S_lm,
                        crystal_outside_grid = coherent_timing.value

                    # The axis was built wider than the bands need, so trim the empty ends if necessary.
                    if crystal_S_lm !== nothing
                        # The 2π/V_cell that makes this the structure function rather than the bare sum.
                        apply_structure_function_prefactor!(crystal_S_lm, lattice.V_cell)
                        built_points = length(crystal_energy_grid.E)
                        crystal_S_lm, crystal_energy_grid =
                            trim_energy_axis(crystal_S_lm, crystal_energy_grid)
                        trimmed_points = built_points - length(crystal_energy_grid.E)
                        trimmed_points == 0 ||
                            println("  Trimmed $(trimmed_points) empty bins of $(built_points) " *
                                    "off the ends of the energy axis.")
                    end
                    push!(stage_times, "coherent f_sq_lm (H(k), diagonalise, project)" => coherent_timing.time)

                    # The same molecules added up without any mixing, which is what the crystal
                    # states must sum back to. Cheap, because contracting in lm space needs no grid.
                    crystal_incoherent_f_sq_lm = nothing
                    if need_incoherent_reference
                        reference_timing = @timed compute_incoherent_unit_cell_f_sq_lm(
                            rotated_R, basis, q_grid_invA, theta_grid, phi_grid, l_max;
                            need_grid = false)
                        crystal_incoherent_f_sq_lm = reference_timing.value[1]
                        push!(stage_times, "incoherent reference f_sq_lm" => reference_timing.time)
                    end

                    print_stage_timings("Crystal stage timings", stage_times)

                    # Show what the crystal did to the monomer energies, at least at the gamma point.
                    print_gamma_point_shifts(
                        basis, lattice, crystal_couplings, crystal_long_range, T)

                    # Check that ∫dE S_lm = (2π/V_cell) Σ_n f²_{n,uc;lm}, with no averaging.
                    if crystal_energy_grid !== nothing
                        println("\n" * describe_energy_grid(crystal_energy_grid))

                        # Make sure that no point fell outside the energy axis.
                        crystal_outside_grid == 0 ||
                            println("  WARNING: $(crystal_outside_grid) states fell outside the " *
                                    "energy axis and were pinned to its ends. Decrease sweep size.")
                        integrated = integrate_energy(crystal_S_lm, crystal_energy_grid)
                        # S_lm carries the 2π/V_cell, so the states it integrates back to need it too.
                        summed = dropdims(sum(crystal_state_f_sq_lm, dims = 1), dims = 1) .*
                                 (T(2 * π) / lattice.V_cell)
                        residual = maximum(abs, integrated .- summed) / maximum(abs, summed)
                        println("  sum rule, ∫dE S_lm(q, E) against (2π/V_cell) Σ_n f²_lm(q): " *
                                string(round(residual, sigdigits = 3)) * " relative.")
                    end
                    println()

                    crystal_results = (
                        order = "coherent",
                        basis = basis,
                        lattice = lattice,
                        dominant_group = dominant_group,
                        dominant_occupancy = dominant_occupancy,
                        band_summary = crystal_band_summary,
                        state_f_sq_lm = crystal_state_f_sq_lm,
                        incoherent_f_sq_lm = crystal_incoherent_f_sq_lm,
                        f_uc = f_uc,
                        I_uc = nothing,
                        S_lm = crystal_S_lm,
                        energy_grid = crystal_energy_grid,
                        couplings = crystal_couplings,
                        cells = cells,
                        q_short_range = q_sr,
                        ewald_parameters = J_parameters === nothing ? nothing :
                            (J = J_parameters, D = D_parameters),
                        long_range = crystal_long_range,
                    )
                end
                end


                # One file per crystal, either order. Coherent is indexed by band n, which
                # is a mixture of monomer transitions and so has no per-transition decomposition;
                # incoherent is indexed by monomer transition. The layout is otherwise the same, so
                # the plotting does not care which it is beyond reading crystal_order.
                if crystal_results !== nothing
                    crystal_output_dir = joinpath(mol_output_dir, "crystal")
                    mkpath(crystal_output_dir)

                    coherent_basis = crystal_results.basis

                    coherent_path = joinpath(crystal_output_dir, "crystal_form_factors$(type_suffix).h5")
                    h5open(coherent_path, "w") do io
                        # Which order (coherent or incoherent) produced this, so a reader knows what the first axis indexes.
                        write(io, "crystal_order", crystal_results.order)
                        write_quantity(io, "V_cell_A3", crystal_results.lattice.V_cell, "angstrom^3")

                        # Coherent: bands n, ordered by ascending energy at each q.
                        # Incoherent: monomer transitions.
                        write(io, "f_sq_lm", crystal_results.state_f_sq_lm)

                        # The same molecules with the mixing switched off, indexed by monomer
                        # transition s. Summing it over s must give back the sum of f_sq_lm over n.
                        if crystal_results.incoherent_f_sq_lm !== nothing
                            write(io, "f_sq_lm_incoherent", crystal_results.incoherent_f_sq_lm)
                        end

                        # The structure function, summed over states.
                        if crystal_results.S_lm !== nothing
                            write(io, "S_lm", crystal_results.S_lm) # Å^-3 eV^-1.
                            HDF5.attributes(io["S_lm"])["units"] = "angstrom^-3 eV^-1"
                            write_quantity(io, "E_grid_eV", crystal_results.energy_grid.E, "eV")
                            # The width of the broadening function, used in place of the delta.
                            write(io, "sigma_E_eV", crystal_results.energy_grid.sigma)
                        end

                        # Coherent amplitudes and incoherent intensities have different names.
                        if crystal_results.f_uc !== nothing || crystal_results.I_uc !== nothing
                            if crystal_results.f_uc !== nothing
                                write(io, "f_uc", crystal_results.f_uc)
                            else
                                write(io, "I_uc", crystal_results.I_uc)
                            end
                            write_quantity(io, "theta_grid", theta_grid, "rad") # Radians.
                            write_quantity(io, "phi_grid", phi_grid, "rad") # Radians.
                        end
                        if crystal_results.band_summary !== nothing
                            write(io, "Delta_E_band_min_eV", crystal_results.band_summary[:, 1])
                            write(io, "Delta_E_band_max_eV", crystal_results.band_summary[:, 2])
                            write(io, "Delta_E_band_mean_eV", crystal_results.band_summary[:, 3])
                        end

                        # Unperturbed monomer energies are present for both crystal orders. The
                        # environment shifts belong only to the coherent Frenkel Hamiltonian.
                        write(io, "Delta_E_monomer_eV", coherent_basis.Delta_E_monomer)
                        if crystal_results.order == "coherent"
                            # D_{λ_iλ_j′}, zero unless λ and λ' sit on the same molecule.
                            write(io, "D_eV", coherent_basis.D_couplings)
                        end

                        # Save the lattice vectors and image translations.
                        write(io, "lattice_A", Matrix{T}(crystal_results.lattice.direct))
                        write(io, "tau_A",
                              reduce(hcat, [image.tau for image in coherent_basis.images]))
                        write(io, "image_of", coherent_basis.image_of)
                        # Needed alongside image_of to read D's blocks back.
                        write(io, "transition_of", coherent_basis.transition_of)

                        # Save everything needed to rebuild H(k), and hence ΔE_n(k) and C(k), at any k
                        # later, using the stored J_deltaR which includes the self subtraction under Ewald:
                        #
                        #     H(k) = diag(ΔE_λ_i) + D + Σ_ΔR J_deltaR(ΔR) exp(i k . ΔR) + J_bar^LR(k).
                        if crystal_results.couplings !== nothing
                            couplings_group = create_group(io, "couplings")
                            write(couplings_group, "method", coupling_method)
                            write(couplings_group, "Delta_R_A", reduce(hcat, crystal_results.cells.Delta_R)) # Å.
                            # The real-space sum over ΔR. Under "ewald" this is the short-range half
                            # alone, with the long-range self interaction already subtracted at ΔR = 0.
                            # Under "direct" it is the whole coupling, and there is no long_range group.
                            write(couplings_group, "J_deltaR_eV", crystal_results.couplings.J_deltaR)
                            HDF5.attributes(couplings_group["J_deltaR_eV"])["units"] = "eV"

                            if crystal_results.long_range !== nothing
                                long_range = crystal_results.long_range
                                splits = crystal_results.ewald_parameters
                                # The reciprocal-space half, J_bar^LR(k). It depends on k, so what is saved
                                # is the ingredients that rebuild it rather than a table over ΔR.
                                # Everything here belongs to J's split, which is the one the table
                                # was built at. D's own cutoffs are saved below it.
                                long_range_group = create_group(couplings_group, "long_range")
                                write(long_range_group, "eta_J", splits.J.eta) # Å^-1.
                                write(long_range_group, "R_max_J", splits.J.R_max) # Å.
                                write(long_range_group, "G_max_J", splits.J.G_max) # Å^-1.
                                write(long_range_group, "eta_D", splits.D.eta) # Å^-1.
                                write(long_range_group, "R_max_D", splits.D.R_max) # Å.
                                write(long_range_group, "G_max_D", splits.D.G_max) # Å^-1.
                                # The requested error both splits were pinned to. Not the medium
                                # permittivity below.
                                write(long_range_group, "epsilon_tot", splits.J.epsilon_tot)
                                # The permittivity of the medium around the sample, scaling the Q = 0 term.
                                write(long_range_group, "epsilon_b", long_range.epsilon_b)
                                write(long_range_group, "G_vectors", reduce(hcat, long_range.G_vectors)) # Å^-1.
                                write(long_range_group, "f_table", long_range.f_table)
                                write(long_range_group, "q_step", long_range.q_step) # Å^-1.
                                write(long_range_group, "l_max_lr", long_range.l_max_lr)
                                write(long_range_group, "dipole_slopes", long_range.dipole_slopes) # Å.
                            end
                        end

                        # What went in.
                        write(io, "transition_indices", collect(transition_indices))
                        write(io, "disorder_group", crystal_results.dominant_group)
                        write(io, "disorder_group_occupancy", crystal_results.dominant_occupancy)
                        write_quantity(io, "q_grid", q_grid, "keV") # keV.
                    end

                    println("$(titlecase(crystal_results.order)) crystal form factor saved to $(coherent_path).")
                end

            end
        end
        catch error
            @error "Crystal form factor computation failed" molecule=mol_num exception=(error, catch_backtrace())
            push!(failures, string(mol_num))
            continue
        end
        result_path = joinpath(mol_output_dir, "crystal", "crystal_form_factors$(type_suffix).h5")
        h5open(result_path, "r+") do io
            HDF5.attributes(io)["stage_signature"] = job["systems"][findfirst(s -> s["number"] == mol_num, job["systems"])]["form_factor_signature"]
            HDF5.attributes(io)["completed"] = true
        end
        if use_gpu
            GC.gc(true)
            CUDA.synchronize()
            CUDA.reclaim()
        end
    end
    return failures
end
