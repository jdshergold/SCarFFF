# This script computes the form factor for a specified molecule.
# Supports spherical, FFT, and Cartesian grid methods.

using HDF5
using ArgParse
using BenchmarkTools
using CSV
using DataFrames
using CUDA
using JSON
using Quaternionic

# Import everything that we need from SCarFFF.
using SCarFFF: compute_cartesian_form_factor, compute_fft_form_factor, compute_spherical_form_factor, compute_rates, compute_rates_by_orientation, construct_crystal_f_lm_tensors, combine_crystal_rate_grids

function parse_commandline()::Dict{String, Any}
    """
    Parse the command line arguments and return them as a dictionary.

    # Arguments:
    - None.

    # Returns:
    - args::Dict{String, Any}: The parsed command line arguments.
    """

    s = ArgParseSettings(
        description = "Compute spherical form factors for molecular transitions."
    )

    @add_arg_table! s begin
        "--smiles"
            help = "SMILES string for a single molecule. Mutually exclusive with --csv-file, --cif-file, and --cif-dir."
            default = nothing
        "--csv-file"
            help = "Path to CSV file containing SMILES strings. Mutually exclusive with --smiles, --cif-file, and --cif-dir."
            default = nothing
        "--cif-file"
            help = "Path to a CIF file describing a crystal (spherical method only). Mutually exclusive with --smiles, --csv-file, and --cif-dir."
            default = nothing
        "--cif-dir"
            help = "Path to a directory of CIF files (spherical method only). Mutually exclusive with --smiles, --csv-file, and --cif-file."
            default = nothing
        "--crystal-mode"
            help = "Whether to process crystals rather than isolated molecules. Only supported for the spherical method."
            action = :store_true
        "--method"
            help = "Computation method for the form factor. Options: 'spherical', 'fft', or 'cartesian'."
            default = "spherical"
        "--q-max"
            help = "Maximum q value in keV."
            arg_type = Float64
            default = 10.0
        "--N-q"
            help = "Number of q grid points (spherical method)."
            arg_type = Int
            default = 101
        "--N-theta"
            help = "Number of theta grid points (spherical method)."
            arg_type = Int
            default = 101
        "--N-phi"
            help = "Number of phi grid points (spherical method)."
            arg_type = Int
            default = 101
        "--l-max"
            help = "Maximum angular momentum quantum number (spherical method)."
            arg_type = Int
            default = 18
        "--threshold"
            help = "Threshold for discarding small tensor entries. For spherical: |W/W_max| < threshold. For Cartesian: |M_ij/M_max| < threshold. Set to 0.0 to disable."
            arg_type = Float64
            default = 1e-6
        "--compute-mode"
            help = "Comma-separated list of outputs to compute/save. Spherical options: form_factor, R_tensor, f_lm_tensor. Cartesian options: V_only, form_factor, both."
            default = "form_factor,R_tensor"
        "--q-lim"
            help = "q-space limits in keV, comma-separated (e.g. 10,10,10) (FFT method)."
            default = "10,10,10"
        "--q-res"
            help = "q-space resolution in keV, comma-separated (e.g. 0.1,0.1,0.1) (FFT method)."
            default = "0.1,0.1,0.1"
        "--qx-grid"
            help = "qx grid specification (min,max,N) in keV (Cartesian method)."
            default = "-10,10,101"
        "--qy-grid"
            help = "qy grid specification (min,max,N) in keV (Cartesian method)."
            default = "-10,10,101"
        "--qz-grid"
            help = "qz grid specification (min,max,N) in keV (Cartesian method)."
            default = "-10,10,101"
        "--check-parseval"
            help = "Whether to check Parseval's theorem holds (FFT method)."
            action = :store_true
        "--transition-indices"
            help = "Transitions to compute the form factor for. Can be 'all' or comma-separated indices like '1,2,3,4'. The first excited state is index 1."
            default = "1"
        "--force-recomputation"
            help = "Whether to force the recomputation of Gaunt coefficients and the A tensor even if they exist."
            action = :store_true
        "--benchmark"
            help = "Whether to run a benchmark of the computation after running it."
            action = :store_true
        "--precision"
            help = "Floating point precision to use. Options: float64, float32."
            default = "float64"
        "--use-gpu"
            help = "Whether to use the GPU for steps beyond the W tensor."
            action = :store_true
        "--output-dir"
            help = "Output directory for this run (e.g., ../runs/run_name). If not specified, defaults to ../runs/{smiles} for a single molecule."
            default = nothing
        "--compute-rates"
            help = "Whether to compute DM scattering rates after the spherical form factor (spherical method only). Requires f_lm_tensor to be available."
            action = :store_true
        "--m-grid"
            help = "DM mass grid for rate computation: 'min_MeV,max_MeV,N' (log-spaced). E.g. '1.0,1000.0,100'."
            default = "1.0,1000.0,100"
        "--N-rotations"
            help = "Number of detector rotations (n_alpha,n_beta,n_gamma) for rate computation. E.g. '12,6,12'. If omitted, uses identity rotation only."
            default = nothing
    end

    return parse_args(s)
end

# Parse command line arguments.
const args = parse_commandline()
const use_gpu = args["use-gpu"]
const method = lowercase(args["method"])

# Check that the specified method is valid.
if !(method in ("spherical", "fft", "cartesian"))
    error("Invalid method '$(method)'. Supported methods are 'spherical', 'fft', and 'cartesian'.")
end

function parse_transition_indices(indices_str::String, td_h5_path::String)::Vector{Int}
    """
    Parse the transition indices string and return a vector of transition indices to compute.

    # Arguments:
    - indices_str::String: The transition indices string. Can be "all" or comma-separated like "1,2,3,4".
    - td_h5_path::String: Path to the TD-DFT HDF5 file to read the total number of transitions.

    # Returns:
    - transition_indices::Vector{Int}: The list of transition indices to compute.
    """

    if lowercase(strip(indices_str)) == "all"
        # Read the TD-DFT file to get the total number of transitions.
        n_transitions = h5open(td_h5_path, "r") do io
            if haskey(io, "transition_matrices")
                length(read(io, "transition_matrices"))
            elseif haskey(io, "energies_ev")
                length(read(io, "energies_ev"))
            else
                count = 0
                while haskey(io, "d_ij_state_$(count + 1)")
                    count += 1
                end
                count
            end
        end
        return collect(1:n_transitions)
    else
        # Parse the comma-separated list.
        return parse.(Int, split(indices_str, ","))
    end
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

function build_crystal_conformer_sets(crystal_metadata, ::Type{T}) where {T<:AbstractFloat}
    """
    Build the conformer-set data structure used for the crystal rate computation.

    # Arguments:
    - crystal_metadata::Dict{String, Any}: The crystal metadata read from JSON.
    - T::Type: The floating point type to use.

    # Returns:
    - conformer_sets::Vector{NamedTuple}: The conformer-set metadata.
    """

    conformer_sets = NamedTuple[]

    for disorder_group in crystal_metadata["disorder_groups"]
        # Group the molecules in this disorder group by their conformer label.
        # This is so that we can accumulate like-molecules later.
        molecules_by_label = Dict{String, Vector{Dict{String, Any}}}()

        for molecule in disorder_group["molecules"]
            label = molecule["conformer_label"]
            # If this is a new label, add a dict entry for it.
            if !haskey(molecules_by_label, label)
                molecules_by_label[label] = Dict{String, Any}[]
            end
            # Add the molecule to the conformer group.
            push!(molecules_by_label[label], molecule)
        end

        # For each conformer, collect the rotations of all its images in this disorder group.
        for label in sort(collect(keys(molecules_by_label)))
            # Initialise the lists.
            proper_rotations = Quaternionic.Rotor{T}[]
            det_rotations = T[]

            # Push the rotations and determinants.
            for molecule in molecules_by_label[label]
                push!(proper_rotations, Quaternionic.rotor(T.(molecule["proper_quaternion"])))
                push!(det_rotations, T(molecule["det_rotation"]))
            end

            # Add the conformer set, name might be "1_A", for disorder group 1, conformer A.
            push!(conformer_sets, (
                name = "$(disorder_group["group"])_$(label)",
                label = label,
                occupancy = T(disorder_group["occupancy"]),
                proper_rotations = proper_rotations,
                det_rotations = det_rotations,
            ))
        end
    end

    return conformer_sets
end

function main()

    # Start timing the computation.
    computation_start = time()

    # Check GPU availability once.
    use_gpu = args["use-gpu"]
    if use_gpu && !CUDA.functional()
        @warn "GPU requested but CUDA is not functional. Falling back to CPU."
        use_gpu = false
    end

    # Extract the common run parameters.
    method = lowercase(args["method"])
    crystal_mode = args["crystal-mode"]
    transition_indices_str = args["transition-indices"]
    force_recomp = args["force-recomputation"]
    run_benchmark = args["benchmark"]
    precision = lowercase(args["precision"])
    if precision == "float32"
        T = Float32
    elseif precision == "float64"
        T = Float64
    else
        error("Invalid precision $(args["precision"]) specified. The supported types are float32 and float64.")
    end

    # Cast variables to correct types.
    typed_zero = zero(T)
    type_suffix = T == Float32 ? "_f32" : "_f64"

    # Check that crystal mode isn't asked for outside of spherical mode.
    if crystal_mode && method != "spherical"
        error("Crystal mode is only supported for the spherical method.")
    end


    input_modes = [
        args["smiles"] !== nothing,
        args["csv-file"] !== nothing,
        args["cif-file"] !== nothing,
        args["cif-dir"] !== nothing,
    ]

    # Check that exactly one input mode is specified.
    if sum(input_modes) != 1
        error("Specify exactly one of --smiles, --csv-file, --cif-file, or --cif-dir.")
    end

    if crystal_mode && args["smiles"] === nothing && args["csv-file"] === nothing
        # This is the expected crystal input path.
    elseif crystal_mode
        error("Crystal mode cannot be used with --smiles or --csv-file.")
    elseif args["cif-file"] !== nothing || args["cif-dir"] !== nothing
        error("Use --crystal-mode when processing CIF files.")
    end

    # Read the molecules or crystals from the specified input mode.
    entries = []
    if args["csv-file"] !== nothing
        # Construct the path to the CSV file.
        script_dir = dirname(abspath(@__FILE__))
        project_root = dirname(script_dir)
        csv_path = joinpath(project_root, args["csv-file"])

        # Read in the list of molecules.
        df = CSV.read(csv_path, DataFrame)
        entries = [(kind = "molecule", display_name = row.smiles, input_value = row.smiles) for row in eachrow(df)]
        println("Found $(length(entries)) molecules to compute form factors for.\n")
    elseif args["smiles"] !== nothing
        # Single molecule mode.
        entries = [(kind = "molecule", display_name = args["smiles"], input_value = args["smiles"])]
        println("Processing single molecule: $(args["smiles"])\n")
    elseif args["cif-dir"] !== nothing
        # Batch crystal mode.
        script_dir = dirname(abspath(@__FILE__))
        project_root = dirname(script_dir)
        cif_dir = joinpath(project_root, args["cif-dir"])
        cif_paths = sort(filter(path -> endswith(lowercase(path), ".cif"), readdir(cif_dir; join=true)))
        if isempty(cif_paths)
            error("No CIF files found in $(cif_dir).")
        end
        entries = [(kind = "crystal", display_name = splitext(basename(cif_path))[1], input_value = cif_path) for cif_path in cif_paths]
        println("Found $(length(entries)) CIF files to compute form factors for.\n")
    elseif args["cif-file"] !== nothing
        # Single crystal mode.
        script_dir = dirname(abspath(@__FILE__))
        project_root = dirname(script_dir)
        cif_path = joinpath(project_root, args["cif-file"])
        entries = [(kind = "crystal", display_name = splitext(basename(cif_path))[1], input_value = cif_path)]
        println("Processing single crystal: $(args["cif-file"])\n")
    else
        error("Specify exactly one of --smiles, --csv-file, --cif-file, or --cif-dir.")
    end

    # Construct the path to the run directory.
    if args["output-dir"] !== nothing
        run_directory = args["output-dir"]
    # If no run name is provided, default to "batch_run".
    elseif args["csv-file"] !== nothing || args["cif-dir"] !== nothing
        run_directory = "../runs/batch_run"
    elseif args["smiles"] !== nothing
        run_directory = "../runs/$(args["smiles"])"
    else
        run_directory = "../runs/$(entries[1].display_name)"
    end

    # Process each molecule or crystal.
    for (mol_num, entry) in enumerate(entries)
        mol_output_dir = joinpath(run_directory, string(mol_num))

        println("="^50)
        println("Computing form factor for $(entry.kind) $mol_num of $(length(entries)), with label $(entry.display_name).")
        println("="^50)
        println()

        # Check if the TD-DFT calculation failed for this molecule or crystal.
        failed_marker = joinpath(mol_output_dir, ".tddft_failed")
        if isfile(failed_marker)
            @warn "TD-DFT was marked as failed for entry $mol_num. Skipping."
            continue
        end

        # Check if TD-DFT or crystal data is missing.
        if crystal_mode
            crystal_metadata_path = joinpath(mol_output_dir, "crystal_metadata.json")
            if !isfile(crystal_metadata_path)
                @warn "No crystal metadata found at $(crystal_metadata_path). Skipping entry $mol_num."
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
            transition_indices = parse_transition_indices(transition_indices_str, td_h5)
            transition_index = transition_indices[1]
            transition_suffix = ordinal_suffix(transition_index)
        end

        if method == "spherical"
            # Parse spherical-specific parameters.
            q_max = args["q-max"]
            N_q = args["N-q"]
            N_theta = args["N-theta"]
            N_phi = args["N-phi"]
            l_max = args["l-max"]
            threshold_val = T(args["threshold"])
            # Parse the compute-mode as a comma-separated list of outputs to compute/save.
            valid_spherical_modes = Set(["form_factor", "R_tensor", "f_lm_tensor"])
            compute_modes = Set(strip.(split(args["compute-mode"], ",")))
            for mode in compute_modes
                if !(mode in valid_spherical_modes)
                    error("Invalid compute-mode '$(mode)'. Valid spherical options are: form_factor, R_tensor, f_lm_tensor.")
                end
            end
            need_grid = "form_factor" in compute_modes
            need_R = "R_tensor" in compute_modes
            need_flm = "f_lm_tensor" in compute_modes
            compute_rates_flag = args["compute-rates"]
            if compute_rates_flag
                need_flm = true  # f_lm tensor is required for rate computation.
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
                transition_indices = parse_transition_indices(transition_indices_str, first_td_h5)
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
                conformer_f_lm = Array{T, 3}[]

                # Compute the spherical form factor for each conformer group.
                for conformer_label in conformer_labels
                    td_h5 = joinpath(mol_output_dir, "conformers", conformer_label, "td_dft_results$(type_suffix).h5")
                    if !isfile(td_h5)
                        error("No conformer TD-DFT results found at $(td_h5).")
                    end

                    # Get the form factor for this conformer.
                    R_tensor, f_s, f_lm, transition_energies_eV = compute_spherical_form_factor(
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
                        need_flm=true
                    )

                    push!(conformer_f_lm, f_lm)
                    conformer_results[conformer_label] = (
                        R_tensor = R_tensor,
                        f_s = f_s,
                        f_lm = f_lm,
                        transition_energies_eV = transition_energies_eV,
                    )

                    # Write the results to HDF5 for this conformer.
                    for (batch_idx, transition_idx) in enumerate(transition_indices)
                        transition_output_dir = joinpath(mol_output_dir, "conformers", conformer_label, string(transition_idx))
                        mkpath(transition_output_dir)

                        transition_energy = transition_energies_eV[batch_idx]
                        output_path = joinpath(transition_output_dir, "fs_grid$(type_suffix).h5")
                        h5open(output_path, "w") do io
                            if f_s !== nothing
                                f_s_slice = f_s[batch_idx, :, :, :]
                                write(io, "f_s", f_s_slice)
                                write(io, "theta_grid", theta_grid)
                                write(io, "phi_grid", phi_grid)
                            end
                            if R_tensor !== nothing
                                R_slice = R_tensor[batch_idx, :, :]
                                write(io, "R_tensor", R_slice)
                            end
                            if f_lm !== nothing
                                f_lm_slice = f_lm[batch_idx, :, :]
                                write(io, "f_lm", f_lm_slice)
                            end
                            write(io, "q_grid", q_grid)
                            write(io, "transition_index", transition_idx)
                            write(io, "transition_energy_eV", transition_energy)
                        end
                    end
                end

                # Now compute the rates for each conformer group.
                m_grid = nothing
                N_rotations = nothing
                if compute_rates_flag

                    # Construct the mass grid.
                    m_vals_str = split(args["m-grid"], ",")
                    m_min = parse(Float64, m_vals_str[1])
                    m_max = parse(Float64, m_vals_str[2])
                    n_masses = parse(Int, m_vals_str[3])
                    m_grid = T.(10 .^ range(log10(m_min), log10(m_max), n_masses))

                    # Construct the rotation grid.
                    N_rotations_arg = args["N-rotations"]
                    N_rotations = if N_rotations_arg !== nothing
                        n = parse.(Int, split(N_rotations_arg, ","))
                        (n[1], n[2], n[3])
                    else
                        nothing
                    end

                    # Compute the rates for each conformer.
                    for conformer_label in conformer_labels
                        conformer_rate_results = compute_rates(
                            conformer_results[conformer_label].f_lm,
                            q_grid,
                            m_grid,
                            T.(conformer_results[conformer_label].transition_energies_eV),
                            N_rotations
                        )

                        # Save the results to disk.
                        conformer_output_dir = joinpath(mol_output_dir, "conformers", conformer_label)
                        mkpath(conformer_output_dir)

                        mchi_vals = T[row.mchi_MeV for row in conformer_rate_results]
                        rate_max_vals = T[row.rate_max for row in conformer_rate_results]
                        rate_min_vals = T[row.rate_min for row in conformer_rate_results]
                        rate_mean_vals = T[row.rate_mean for row in conformer_rate_results]
                        qmax_keV = q_grid[end]

                        rates_h5_path = joinpath(conformer_output_dir, "scattering_rates$(type_suffix).h5")
                        h5open(rates_h5_path, "w") do io
                            write(io, "mchi_MeV", mchi_vals)
                            write(io, "rate_max", rate_max_vals)
                            write(io, "rate_min", rate_min_vals)
                            write(io, "rate_mean", rate_mean_vals)
                            write(io, "qmax_keV", qmax_keV)
                        end
                    end
                end

                # Turn the individual conformer results into the crystal results.
                conformer_sets = build_crystal_conformer_sets(crystal_metadata, T)

                # set_f_lm is the one aggregated over one conformer group (e.g. 1_A), the other is aggregated over all groups.
                set_f_lm, aggregate_f_lm = construct_crystal_f_lm_tensors(conformer_labels, conformer_f_lm, conformer_sets)
                conformer_set_occupancies = T[conformer_set.occupancy for conformer_set in conformer_sets]

                for (batch_idx, transition_idx) in enumerate(transition_indices)
                    transition_output_dir = joinpath(mol_output_dir, "crystal", string(transition_idx))
                    mkpath(transition_output_dir)

                    # Allocate the output tensors.
                    # Conformer tensor is e.g. "A", "B", etc.
                    # Conformer set tensor is e.g. "1_A", "1_B", etc.
                    conformer_tensor = Array{T}(undef, length(conformer_labels), size(aggregate_f_lm, 2), size(aggregate_f_lm, 3))
                    conformer_energies = Vector{T}(undef, length(conformer_labels))
                    conformer_set_tensor = Array{T}(undef, length(set_f_lm), size(aggregate_f_lm, 2), size(aggregate_f_lm, 3))
                    conformer_set_energies = Vector{T}(undef, length(set_f_lm))
                    conformer_set_names = String[]
                    conformer_set_labels = String[]

                    # Fill the individual conformer tensors.
                    for (conformer_idx, conformer_label) in enumerate(conformer_labels)
                        conformer_tensor[conformer_idx, :, :] = conformer_results[conformer_label].f_lm[batch_idx, :, :]
                        conformer_energies[conformer_idx] = conformer_results[conformer_label].transition_energies_eV[batch_idx]
                    end

                    # Fill the conformer set tensors.
                    for (set_idx, conformer_set) in enumerate(conformer_sets)
                        conformer_set_tensor[set_idx, :, :] = set_f_lm[set_idx][batch_idx, :, :]
                        conformer_set_energies[set_idx] = conformer_results[conformer_set.label].transition_energies_eV[batch_idx]
                        push!(conformer_set_names, conformer_set.name)
                        push!(conformer_set_labels, conformer_set.label)
                    end

                    # Save the results to disk.
                    output_path = joinpath(transition_output_dir, "fs_grid$(type_suffix).h5")
                    h5open(output_path, "w") do io
                        write(io, "f_lm", aggregate_f_lm[batch_idx, :, :])
                        write(io, "conformer_f_lm", conformer_tensor)
                        write(io, "conformer_labels", conformer_labels)
                        write(io, "conformer_transition_energies_eV", conformer_energies)
                        write(io, "conformer_set_f_lm", conformer_set_tensor)
                        write(io, "conformer_set_names", conformer_set_names)
                        write(io, "conformer_set_labels", conformer_set_labels)
                        write(io, "conformer_set_occupancies", conformer_set_occupancies)
                        write(io, "conformer_set_transition_energies_eV", conformer_set_energies)
                        write(io, "q_grid", q_grid)
                        write(io, "transition_index", transition_idx)
                    end
                end

                if compute_rates_flag
                    conformer_rate_grids = Array{T, 2}[]
                    # Compute the rates for each conformer set (e.g. 1_A, 1_B, etc.)
                    for (set_idx, conformer_set) in enumerate(conformer_sets)
                        transition_energies_eV = T.(conformer_results[conformer_set.label].transition_energies_eV)
                        rate_grid = compute_rates_by_orientation(set_f_lm[set_idx], q_grid, m_grid, transition_energies_eV, N_rotations)
                        push!(conformer_rate_grids, rate_grid)
                    end

                    # Combine the sets into a total crystal rate grid.
                    rate_results = combine_crystal_rate_grids(conformer_rate_grids, conformer_set_occupancies, m_grid)

                    # Save the results to disk.
                    crystal_output_dir = joinpath(mol_output_dir, "crystal")
                    mkpath(crystal_output_dir)

                    mchi_vals = T[row.mchi_MeV for row in rate_results]
                    rate_max_vals = T[row.rate_max for row in rate_results]
                    rate_min_vals = T[row.rate_min for row in rate_results]
                    rate_mean_vals = T[row.rate_mean for row in rate_results]
                    qmax_keV = q_grid[end]

                    rates_h5_path = joinpath(crystal_output_dir, "scattering_rates$(type_suffix).h5")
                    h5open(rates_h5_path, "w") do io
                        write(io, "mchi_MeV", mchi_vals)
                        write(io, "rate_max", rate_max_vals)
                        write(io, "rate_min", rate_min_vals)
                        write(io, "rate_mean", rate_mean_vals)
                        write(io, "qmax_keV", qmax_keV)
                    end
                end
            else
                transition_indices = parse_transition_indices(transition_indices_str, td_h5)
                transition_index = transition_indices[1]
                transition_suffix = ordinal_suffix(transition_index)

                # Print transition info.
                if length(transition_indices) == 1
                    tidx = transition_indices[1]
                    println("Computing the spherical form factor for the $(tidx)$(ordinal_suffix(tidx)) transition of the molecule with SMILES $(entry.display_name) up to maximum angular mode l = $(l_max).")
                else
                    println("Computing the spherical form factor for $(length(transition_indices)) transitions of the molecule with SMILES $(entry.display_name) up to maximum angular mode l = $(l_max).")
                    println("Transition indices: $(transition_indices)")
                end
                println("Grid: q ∈ [0, $(q_max)] keV with $(N_q) points,")
                println("      θ ∈ [0, π] with $(N_theta) points,")
                println("      φ ∈ [0, 2π] with $(N_phi) points.")

                # Compute the spherical form factor.
                R_tensor, f_s, f_lm, transition_energies_eV = compute_spherical_form_factor(
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
                    need_flm=need_flm
                )

                rate_results = nothing

                # Compute DM scattering rates if requested.
                if compute_rates_flag && f_lm !== nothing
                    m_vals_str = split(args["m-grid"], ",")
                    m_min = parse(Float64, m_vals_str[1])
                    m_max = parse(Float64, m_vals_str[2])
                    n_masses = parse(Int, m_vals_str[3])
                    m_grid = T.(10 .^ range(log10(m_min), log10(m_max), n_masses))

                    N_rotations_arg = args["N-rotations"]
                    N_rotations = if N_rotations_arg !== nothing
                        n = parse.(Int, split(N_rotations_arg, ","))
                        (n[1], n[2], n[3])
                    else
                        nothing
                    end

                    rate_results = compute_rates(f_lm, q_grid, m_grid, T.(transition_energies_eV), N_rotations)
                end

                # Run the benchmark. Testing only.
                if run_benchmark
                    println("\nRunning benchmark...")
                    @btime compute_spherical_form_factor(
                        $q_grid,
                        $theta_grid,
                        $phi_grid,
                        $l_max,
                        $td_h5,
                        transition_indices=$transition_indices,
                        force_recomputation=$force_recomp,
                        threshold=$threshold_val,
                        use_gpu=$use_gpu,
                        need_grid=$need_grid,
                        need_R=$need_R,
                        need_flm=$need_flm
                    )
                end

                # Save the results to disk.
                # Loop over each computed transition and save them separately.
                for (batch_idx, transition_idx) in enumerate(transition_indices)
                    # Create the directory structure.
                    transition_output_dir = joinpath(mol_output_dir, "spherical", "transition_$(transition_idx)")
                    mkpath(transition_output_dir)

                    transition_energy = transition_energies_eV[batch_idx]

                    # Save to HDF5.
                    output_path = joinpath(transition_output_dir, "fs_grid$(type_suffix).h5")
                    h5open(output_path, "w") do io
                        if f_s !== nothing
                            f_s_slice = f_s[batch_idx, :, :, :]
                            write(io, "f_s", f_s_slice)
                            write(io, "theta_grid", theta_grid)
                            write(io, "phi_grid", phi_grid)
                        end
                        if R_tensor !== nothing
                            R_slice = R_tensor[batch_idx, :, :]
                            write(io, "R_tensor", R_slice)
                        end
                        if f_lm !== nothing
                            f_lm_slice = f_lm[batch_idx, :, :]
                            write(io, "f_lm", f_lm_slice)
                        end
                        write(io, "q_grid", q_grid)
                        write(io, "transition_index", transition_idx)
                        write(io, "transition_energy_eV", transition_energy)
                    end
                end

                if rate_results !== nothing
                    spherical_output_dir = joinpath(mol_output_dir, "spherical")
                    mkpath(spherical_output_dir)

                    mchi_vals = T[row.mchi_MeV for row in rate_results]
                    rate_max_vals = T[row.rate_max for row in rate_results]
                    rate_min_vals = T[row.rate_min for row in rate_results]
                    rate_mean_vals = T[row.rate_mean for row in rate_results]
                    qmax_keV = q_grid[end]

                    rates_h5_path = joinpath(spherical_output_dir, "scattering_rates$(type_suffix).h5")
                    h5open(rates_h5_path, "w") do io
                        write(io, "mchi_MeV", mchi_vals)
                        write(io, "rate_max", rate_max_vals)
                        write(io, "rate_min", rate_min_vals)
                        write(io, "rate_mean", rate_mean_vals)
                        write(io, "qmax_keV", qmax_keV)
                    end
                end
            end
        elseif method == "fft"
            # Parse FFT-specific parameters.
            q_lim = parse.(T, split(args["q-lim"], ","))
            q_res = parse.(T, split(args["q-res"], ","))
            check_parseval = args["check-parseval"]

            # Define the momentum grid.
            N_qx = round(Int, 2 * q_lim[1] / q_res[1]) + 1
            N_qy = round(Int, 2 * q_lim[2] / q_res[2]) + 1
            N_qz = round(Int, 2 * q_lim[3] / q_res[3]) + 1

            # Create the grid vectors.
            qx_grid = collect(range(-T(q_lim[1]), T(q_lim[1]), length=N_qx))
            qy_grid = collect(range(-T(q_lim[2]), T(q_lim[2]), length=N_qy))
            qz_grid = collect(range(-T(q_lim[3]), T(q_lim[3]), length=N_qz))

            if length(transition_indices) == 1
                println("Computing the FFT form factor for the $(transition_index)$(transition_suffix) transition of the molecule with SMILES $(entry.display_name).")
            else
                println("Computing the FFT form factor for $(length(transition_indices)) transitions of the molecule with SMILES $(entry.display_name).")
            end
            println("Grid: qx ∈ [$(qx_grid[1]), $(qx_grid[end])] keV with $(N_qx) points,")
            println("      qy ∈ [$(qy_grid[1]), $(qy_grid[end])] keV with $(N_qy) points,")
            println("      qz ∈ [$(qz_grid[1]), $(qz_grid[end])] keV with $(N_qz) points.")

            # Compute the FFT form factor.
            form_factors, transition_densities, r_lim, transition_energies_eV = compute_fft_form_factor(
                qx_grid,
                qy_grid,
                qz_grid,
                td_h5,
                transition_indices=transition_indices,
                check_parseval=check_parseval,
                use_gpu=use_gpu
            )

            # Run the benchmark. Testing only.
            if run_benchmark
                println("\nRunning benchmark...")
                @btime compute_fft_form_factor(
                    $qx_grid,
                    $qy_grid,
                    $qz_grid,
                    $td_h5,
                    transition_indices=$transition_indices,
                    check_parseval=false,
                    use_gpu=$use_gpu
                )
            end

            # Save the results to disk. Loop over each computed transition and save them separately.
            for (batch_idx, transition_idx) in enumerate(transition_indices)
                transition_output_dir = joinpath(mol_output_dir, "fft", "transition_$(transition_idx)")
                mkpath(transition_output_dir)

                form_factor = form_factors[batch_idx, :, :, :]
                transition_density = transition_densities[batch_idx, :, :, :]
                transition_energy_eV = transition_energies_eV[batch_idx]

                output_path = joinpath(transition_output_dir, "fs_grid$(type_suffix).h5")
                h5open(output_path, "w") do io
                    write(io, "form_factor", form_factor)
                    write(io, "transition_density", transition_density)
                    write(io, "qx_grid", qx_grid)
                    write(io, "qy_grid", qy_grid)
                    write(io, "qz_grid", qz_grid)
                    write(io, "r_lim", r_lim)
                    write(io, "transition_index", transition_idx)
                    write(io, "transition_energy_eV", transition_energy_eV)
                end
            end
        elseif method == "cartesian"
            # Parse Cartesian-specific parameters.
            qx_spec = parse.(T, split(args["qx-grid"], ","))
            qy_spec = parse.(T, split(args["qy-grid"], ","))
            qz_spec = parse.(T, split(args["qz-grid"], ","))
            threshold_val = T(args["threshold"])
            compute_mode_str = lowercase(args["compute-mode"])

            # Determine what to compute based on compute_mode_str.
            if compute_mode_str == "v_only"
                need_grid, need_V = false, true
            elseif compute_mode_str == "form_factor"
                need_grid, need_V = true, false
            elseif compute_mode_str == "both"
                need_grid, need_V = true, true
            else
                error("Invalid compute-mode $(args["compute-mode"]) for Cartesian method. The options are \"form_factor\", \"V_only\", and \"both\".")
            end

            # Extract (min, max, N) from each specification.
            qx_min, qx_max, N_qx = qx_spec[1], qx_spec[2], round(Int, qx_spec[3])
            qy_min, qy_max, N_qy = qy_spec[1], qy_spec[2], round(Int, qy_spec[3])
            qz_min, qz_max, N_qz = qz_spec[1], qz_spec[2], round(Int, qz_spec[3])

            # Create the grid vectors.
            qx_grid = collect(range(T(qx_min), T(qx_max), length=N_qx))
            qy_grid = collect(range(T(qy_min), T(qy_max), length=N_qy))
            qz_grid = collect(range(T(qz_min), T(qz_max), length=N_qz))

            if length(transition_indices) == 1
                println("Computing the Cartesian form factor for the $(transition_index)$(transition_suffix) transition of the molecule with SMILES $(entry.display_name).")
            else
                println("Computing the Cartesian form factor for $(length(transition_indices)) transitions of the molecule with SMILES $(entry.display_name).")
            end
            println("Grid: qx ∈ [$(qx_grid[1]), $(qx_grid[end])] keV with $(N_qx) points,")
            println("      qy ∈ [$(qy_grid[1]), $(qy_grid[end])] keV with $(N_qy) points,")
            println("      qz ∈ [$(qz_grid[1]), $(qz_grid[end])] keV with $(N_qz) points.")

            # Compute the Cartesian form factor.
            V_tensors, form_factor, transition_energies_eV = compute_cartesian_form_factor(
                qx_grid,
                qy_grid,
                qz_grid,
                td_h5,
                transition_indices=transition_indices,
                threshold=threshold_val,
                use_gpu=use_gpu,
                need_grid=need_grid,
                need_V=need_V
            )

            # Run the benchmark. Testing only.
            if run_benchmark
                println("\nRunning benchmark...")
                @btime compute_cartesian_form_factor(
                    $qx_grid,
                    $qy_grid,
                    $qz_grid,
                    $td_h5,
                    transition_indices=$transition_indices,
                    threshold=$threshold_val,
                    use_gpu=$use_gpu,
                    need_grid=$need_grid,
                    need_V=$need_V
                )
            end

            # Save the results to disk. Loop over each computed transition and save them separately.
            for (batch_idx, transition_idx) in enumerate(transition_indices)
                transition_output_dir = joinpath(mol_output_dir, "cartesian", "transition_$(transition_idx)")
                mkpath(transition_output_dir)

                transition_energy = transition_energies_eV[batch_idx]

                output_path = joinpath(transition_output_dir, "fs_grid$(type_suffix).h5")
                h5open(output_path, "w") do io
                    if form_factor !== nothing
                        form_factor_slice = form_factor[batch_idx, :, :, :]
                        write(io, "form_factor", form_factor_slice)
                    end
                    if V_tensors !== nothing
                        V_x, V_y, V_z = V_tensors
                        write(io, "V_x", V_x)
                        write(io, "V_y", V_y)
                        write(io, "V_z", V_z)
                    end
                    write(io, "qx_grid", qx_grid)
                    write(io, "qy_grid", qy_grid)
                    write(io, "qz_grid", qz_grid)
                    write(io, "transition_index", transition_idx)
                    write(io, "transition_energy_eV", transition_energy)
                end
            end
        else
            error("Invalid method '$(method)'. Supported methods are 'spherical', 'fft', and 'cartesian'.")
        end

        catch e
            @warn "Form factor computation failed for entry $mol_num: $e. Skipping."
            continue
        end

        println("\n$(uppercasefirst(entry.kind)) $mol_num complete!\n")

        # Release any cached GPU allocations between molecules.
        if use_gpu
            GC.gc(true)  # Force garbage collection.
            CUDA.synchronize()  # Ensure that all GPU operations are complete.
            CUDA.reclaim()  # Then reclaim memory pool.
        end
    end

    # End timing and save to file for bash to read.
    computation_time = time() - computation_start
    timing_file = joinpath(run_directory, ".form_factor_time")
    open(timing_file, "w") do io
        println(io, round(computation_time, digits=3))
    end

    println("="^50)
    println("All form factor calculations complete!")
    println("Results saved to: $run_directory")
    println("="^50)
end

# Run the main function.
main()
