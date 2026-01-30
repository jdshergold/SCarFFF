# This script computes the form factor for a specified molecule.
# Supports spherical, FFT, and Cartesian grid methods.

using HDF5
using ArgParse
using BenchmarkTools
using CSV
using DataFrames
using CUDA

# Import everything that we need from SCarFFF.
using SCarFFF: compute_cartesian_form_factor, compute_fft_form_factor, compute_spherical_form_factor

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
            help = "SMILES string for a single molecule (ignored if --csv-file is provided)."
            default = nothing
        "--csv-file"
            help = "Path to CSV file containing SMILES strings (ignored if processing single molecule)."
            default = nothing
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
            help = "What to compute/save. For spherical: R_only, form_factor, both. For Cartesian: V_only, form_factor, both."
            default = "both"
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

    # Read the molecules from the metadata.csv file, or use a single SMILES.
    molecules = []
    if args["csv-file"] !== nothing
        # Construct the path to the CSV file.
        script_dir = dirname(abspath(@__FILE__))
        project_root = dirname(script_dir)
        csv_path = joinpath(project_root, args["csv-file"])

        # Read in the list of molecules.
        df = CSV.read(csv_path, DataFrame)
        molecules = [(i, row.smiles) for (i, row) in enumerate(eachrow(df))]
        println("Found $(length(molecules)) molecules to compute form factors for.\n")
    elseif args["smiles"] !== nothing
        # Single molecule mode.
        molecules = [(1, args["smiles"])]
        println("Processing single molecule: $(args["smiles"])\n")
    else
        error("Either --smiles or --csv-file must be provided.")
    end

    # Construct the path to the run directory.
    if args["output-dir"] !== nothing
        run_directory = args["output-dir"]
    # If no run name is provided, default to "batch_run".
    elseif args["csv-file"] !== nothing
        run_directory = "../runs/batch_run"
    else
        run_directory = "../runs/$(args["smiles"])"
    end

    # Process each molecule.
    for (mol_num, smiles) in molecules
        mol_output_dir = joinpath(run_directory, string(mol_num))
        td_h5 = joinpath(mol_output_dir, "td_dft_results$(type_suffix).h5")

        println("="^50)
        println("Computing form factor for molecule $mol_num of $(length(molecules)), with SMILES $smiles.")
        println("="^50)
        println()

        # Check if the TD-DFT calculation failed for this molecule.
        failed_marker = joinpath(mol_output_dir, ".tddft_failed")
        if isfile(failed_marker)
            @warn "TD-DFT was marked as failed for molecule $mol_num. Skipping."
            continue
        end

        # Check if TD-DFT results exist.
        if !isfile(td_h5)
            @warn "No TD-DFT results found at $(td_h5). Skipping molecule $mol_num."
            continue
        end

        # Wrap the form factor computation in a try/catch so that a single molecule failing
        # (e.g. due to missing or corrupt data) does not crash the entire batch.
        try

        # Parse the transition indices.
        transition_indices = parse_transition_indices(transition_indices_str, td_h5)
        transition_index = transition_indices[1]
        transition_suffix = transition_index % 10 == 1 && transition_index % 100 != 11 ? "st" :
                            transition_index % 10 == 2 && transition_index % 100 != 12 ? "nd" :
                            transition_index % 10 == 3 && transition_index % 100 != 13 ? "rd" : "th"

        if method == "spherical"
            # Parse spherical-specific parameters.
            q_max = args["q-max"]
            N_q = args["N-q"]
            N_theta = args["N-theta"]
            N_phi = args["N-phi"]
            l_max = args["l-max"]
            threshold_val = T(args["threshold"])
            compute_mode_str = lowercase(args["compute-mode"])
            # Determine what to compute based on compute_mode_str.
            if compute_mode_str == "r_only"
                need_grid, need_R = false, true
            elseif compute_mode_str == "form_factor"
                need_grid, need_R = true, false
            elseif compute_mode_str == "both"
                need_grid, need_R = true, true
            else
                error("Invalid compute-mode $(args["compute-mode"]). The options are \"form_factor\", \"R_only\", and \"both\".")
            end

            # Define the momentum grid.
            q_grid = collect(range(typed_zero, T(q_max), length=N_q))
            theta_grid = collect(range(typed_zero, T(π), length=N_theta))
            phi_grid = collect(range(typed_zero, T(2π), length=N_phi))

            # Print transition info.
            if length(transition_indices) == 1
                tidx = transition_indices[1]
                suffix = tidx % 10 == 1 && tidx % 100 != 11 ? "st" :
                         tidx % 10 == 2 && tidx % 100 != 12 ? "nd" :
                         tidx % 10 == 3 && tidx % 100 != 13 ? "rd" : "th"
                println("Computing the spherical form factor for the $(tidx)$(suffix) transition of the molecule with SMILES $(smiles) up to maximum angular mode l = $(l_max).")
            else
                println("Computing the spherical form factor for $(length(transition_indices)) transitions of the molecule with SMILES $(smiles) up to maximum angular mode l = $(l_max).")
                println("Transition indices: $(transition_indices)")
            end
            println("Grid: q ∈ [0, $(q_max)] keV with $(N_q) points,")
            println("      θ ∈ [0, π] with $(N_theta) points,")
            println("      φ ∈ [0, 2π] with $(N_phi) points.")

            # Compute the spherical form factor.
            R_tensor, f_s, transition_energies_eV = compute_spherical_form_factor(
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
                need_R=need_R
            )

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
                    need_R=$need_R
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
                    write(io, "q_grid", q_grid)
                    write(io, "transition_index", transition_idx)
                    write(io, "transition_energy_eV", transition_energy)
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
                println("Computing the FFT form factor for the $(transition_index)$(transition_suffix) transition of the molecule with SMILES $(smiles).")
            else
                println("Computing the FFT form factor for $(length(transition_indices)) transitions of the molecule with SMILES $(smiles).")
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
                println("Computing the Cartesian form factor for the $(transition_index)$(transition_suffix) transition of the molecule with SMILES $(smiles).")
            else
                println("Computing the Cartesian form factor for $(length(transition_indices)) transitions of the molecule with SMILES $(smiles).")
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
            @warn "Form factor computation failed for molecule $mol_num: $e. Skipping."
            continue
        end

        println("\nMolecule $mol_num complete!\n")

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
