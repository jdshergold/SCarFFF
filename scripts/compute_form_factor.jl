# This script computes the form factor for a specified molecule.
# Supports spherical, FFT, and Cartesian grid methods.

using HDF5
using ArgParse
using CSV
using DataFrames
using CUDA
using JSON
using Quaternionic

# Import everything that we need from SCarFFF.
using StaticArrays
using SCarFFF: compute_cartesian_form_factor, compute_fft_form_factor, compute_spherical_form_factor, compute_rates, compute_rates_by_orientation, construct_crystal_f_lm_tensors, combine_crystal_rate_grids
using SCarFFF.SphericalFormFactor: CrystalImage, build_excitation_basis, build_crystal_lattice,
                                  compute_spherical_form_factor_with_densities,
                                  rotate_R_tensors, rotate_crystal_R_tensors,
                                  compute_coherent_crystal_f_lm,
                                  compute_incoherent_crystal_f_lm, project_f_lm,
                                  enumerate_neighbour_cells, compute_couplings,
                                  compute_crystal_corrections, default_angular_grid,
                                  choose_ewald_parameters, build_ewald_long_range,
                                  compute_ewald_diagonal, subtract_self_term!,
                                  image_translation_span, supercell_radius,
                                  set_diagonal_corrections!,
                                  derive_symmetry_operations, build_stars,
                                  star_reduction_factor, choose_compatible_phi_count
using SCarFFF.SphericalFormFactor.PrecomputeGaunt: precompute_gaunt_coefficients
using SCarFFF.SphericalFormFactor.CrystalLattice: KEV_TO_INV_ANGSTROM
using SCarFFF.SphericalFormFactor.BlochHamiltonian: BlochEigensystem, solve_bloch_hamiltonian!,
                                                    solve_bloch_energies!
using SCarFFF.ThreadChunks: chunk_count, chunk_range
using SCarFFF.StageTimings: print_stage_timings
using Base.Threads
using LinearAlgebra: norm, dot


function sample_band_plane(
        basis, lattice, couplings, long_range, plane::String, n_points::Int, ::Type{T},
    ) where {T<:AbstractFloat}
    """
    Sample the Frenkel exciton band energies E_Ψ(k) over a Cartesian plane of the first Brillouin
    zone, so the axes line up with the form factor slices.

    # Arguments:
    - basis: The localised excitation basis.
    - lattice: The crystal lattice.
    - couplings: The real-space couplings, or nothing.
    - long_range: The Ewald long-range data, or nothing.
    - plane::String: The Cartesian plane to sample: xy, xz or yz.
    - n_points::Int: Samples along each axis.
    - T::Type: The floating point type.

    # Returns:
    - Tuple: The two axes in keV, and the energies with dimensions (n_states, n_points, n_points),
      NaN outside the zone.
    """

    # Get the reciprocal lattice vectors and cell volume.
    b1 = SVector{3, T}(lattice.reciprocal[1, :])
    b2 = SVector{3, T}(lattice.reciprocal[2, :])
    b3 = SVector{3, T}(lattice.reciprocal[3, :])
    a_rows = (SVector{3, T}(lattice.direct[1, :]), SVector{3, T}(lattice.direct[2, :]),
              SVector{3, T}(lattice.direct[3, :]))

    # The zone corners are the eight (±b1 ±b2 ±b3)/2, so their extreme Cartesian components bound it.
    half_widths = zeros(T, 3)
    for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
        corner = (s1 * b1 + s2 * b2 + s3 * b3) / 2
        half_widths .= max.(half_widths, abs.(corner))
    end

    first_axis, second_axis = plane == "xy" ? (1, 2) : plane == "xz" ? (1, 3) : (2, 3)
    axis_a_invA = collect(range(-half_widths[first_axis], half_widths[first_axis], length = n_points))
    axis_b_invA = collect(range(-half_widths[second_axis], half_widths[second_axis], length = n_points))

    n_states = length(basis.energies)
    energies = fill(T(NaN), n_states, n_points, n_points)
    n_cells = couplings === nothing ? 0 : length(couplings.cell_vectors)

    # Time reversal gives E(k) = E(-k) for any crystal, and the axes are symmetric about zero, so the
    # antipode of sample (i, j) is exactly sample (n + 1 - i, n + 1 - j). We get this for free.
    antipode(i, j) = (n_points + 1 - i, n_points + 1 - j)
    to_be_solved = falses(n_points, n_points)
    @inbounds for i in 1:n_points, j in 1:n_points
        # Of each (k, -k) pair, the one reached first does the work and the other copies it later.
        # k = 0 is its own partner, so it has nothing to copy from and must solve itself.
        partner = antipode(i, j)
        to_be_solved[i, j] = partner >= (i, j)
    end

    # Rebuild the Hamiltonian and diagonalise for the energies at each k. Only the energies are ever
    # read, so don't solve for the eigenvectors.
    n_chunks = chunk_count(n_points, nthreads())
    @sync for chunk in 1:n_chunks
        Threads.@spawn begin
            eigensystem = BlochEigensystem(basis, long_range; n_cells = n_cells, vecs = false)
            for i in chunk_range(chunk, n_chunks, n_points)
                for j in 1:n_points
                    to_be_solved[i, j] || continue

                    components = zeros(T, 3)
                    components[first_axis] = axis_a_invA[i]
                    components[second_axis] = axis_b_invA[j]
                    k_vector = SVector{3, T}(components)

                    # Inside the zone means every fractional coordinate within [-1/2, 1/2].
                    inside = all(abs(dot(a_rows[axis], k_vector) / (2 * T(π))) <= T(0.5) + eps(T)
                                 for axis in 1:3)
                    inside || continue

                    solve_bloch_energies!(eigensystem, basis, k_vector, couplings, long_range)
                    @inbounds for state in 1:n_states
                        energies[state, i, j] = eigensystem.energies[state]
                    end
                end
            end
        end
    end

    # Mirror the solved half onto its antipodes. The zone test is symmetric under k -> -k, so a NaN
    # simply carries across and the blank region outside the zone is preserved.
    @inbounds for i in 1:n_points, j in 1:n_points
        to_be_solved[i, j] && continue
        mirror_i, mirror_j = antipode(i, j)
        for state in 1:n_states
            energies[state, i, j] = energies[state, mirror_i, mirror_j]
        end
    end

    axis_a = axis_a_invA ./ T(KEV_TO_INV_ANGSTROM)
    axis_b = axis_b_invA ./ T(KEV_TO_INV_ANGSTROM)

    return (axis_a, axis_b), energies
end


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
        "--crystal-order"
            help = "How to combine monomer form factors into the crystal form factor. 'coherent' solves the Frenkel exciton Bloch problem and mixes the molecular amplitudes with the resulting coefficients. 'incoherent' is the zeroth-order approximation that adds |f|^2 over the rotated images. Options: 'coherent' or 'incoherent'."
            default = "incoherent"
        "--coupling-method"
            help = "How to sum the intermolecular couplings over the lattice (coherent crystal mode). 'ewald' splits the Coulomb kernel into a short-range real-space sum and a long-range reciprocal-space one, both of which converge exponentially, so the error is set by --ewald-epsilon. 'direct' sums the bare 1/r kernel out to --coupling-cutoff, which is only conditionally convergent and is kept as a cross-check. Options: 'ewald' or 'direct'."
            default = "ewald"
        "--ewald-epsilon"
            help = "Requested truncation error for the Ewald sums. Both the real- and reciprocal-space cutoffs are pinned to it, so this is the single accuracy knob."
            arg_type = Float64
            default = 1.0e-3
        "--ewald-eta"
            help = "Fixed Ewald splitting parameter in inverse Angstroms, or 0 to derive the one that minimises the total work. The answer must not depend on it, so setting it by hand is how that gets tested."
            arg_type = Float64
            default = 0.0
        "--ewald-cost-ratio"
            help = "The cost of one reciprocal lattice vector relative to one real-space cell, which sets where the derived Ewald cutoffs balance. Only the ratio matters."
            arg_type = Float64
            default = 1.0
        "--no-dipole-term"
            help = "Drop the Q = 0 term of the Ewald reciprocal sum instead of replacing it with the angular-averaged transition dipole product, which is the conducting boundary condition."
            action = :store_true
        "--coupling-cutoff"
            help = "Real-space cutoff in Angstroms for the intermolecular coupling sum, used only by --coupling-method direct. Under Ewald the cutoff is derived from --ewald-epsilon instead. A cutoff of 0 still couples the molecules within the unit cell, since those sit at zero lattice vector; it just excludes neighbouring cells. Use --no-couplings to switch the correction off entirely."
            arg_type = Float64
            default = 40.0
        "--band-map-plane"
            help = "Sample the band energies E_Psi(k) over a Cartesian plane of the first " *
                   "Brillouin zone and save them alongside the coherent output, for plotting. The " *
                   "planes are the same as the form factor slices: xy is k_z = 0, xz is k_y = 0, " *
                   "yz is k_x = 0. Use 'all' for all three, or 'none' to skip it."
            default = "none"
        "--band-map-points"
            help = "Samples along each axis of the band map plane. Prefer an odd value, so that k = 0 is " *
                   "sampled and the time-reversal pairing has a fixed point rather than a gap."
            arg_type = Int
            default = 101
        "--no-couplings"
            help = "Switch the intermolecular corrections J and D off entirely (coherent crystal mode), leaving the bands flat at the monomer energies and recovering the zeroth-order limit."
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
            help = "Number of theta grid points (spherical method). Leave at 0 to size it from l-max, which is what the projection onto spherical harmonics actually needs."
            arg_type = Int
            default = 0
        "--N-phi"
            help = "Number of phi grid points (spherical method). Leave at 0 to size it from l-max. Prefer an odd value: phi + pi only lands on the grid when this is odd, and the crystal symmetry reduction loses roughly half its operations, time reversal included, when it does not."
            arg_type = Int
            default = 0
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
        # Count the number of transitions in the TD-DFT HDF5 file.
        n_transitions, source = h5open(td_h5_path, "r") do io
            count = 0
            while haskey(io, "X_state_$(count + 1)") && haskey(io, "Y_state_$(count + 1)")
                count += 1
            end
            count > 0 && return count, "X and Y amplitudes"

            if haskey(io, "transition_matrices")
                return length(read(io, "transition_matrices")), "transition_matrices"
            end

            while haskey(io, "d_ij_state_$(count + 1)")
                count += 1
            end
            count > 0 && return count, "d_ij matrices"

            haskey(io, "energies_ev") && return length(read(io, "energies_ev")), "energies_ev"
            return 0, "nothing"
        end

        n_transitions > 0 || error(
            "Found no transitions in $(td_h5_path). Expected X_state_1 and Y_state_1, or one of the " *
            "older transition_matrices or d_ij_state_1 layouts."
        )
        println("Found $(n_transitions) transitions, counted from the $(source).")

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
    crystal_order = lowercase(args["crystal-order"])
    crystal_order in ("coherent", "incoherent") ||
        error("Invalid crystal order '$(args["crystal-order"])'. The supported options are 'coherent' and 'incoherent'.")
    coupling_method = lowercase(args["coupling-method"])
    coupling_method in ("ewald", "direct") ||
        error("Invalid coupling method '$(args["coupling-method"])'. The supported options are 'ewald' and 'direct'.")
    coupling_cutoff = args["coupling-cutoff"]
    coupling_cutoff >= 0 || error("The coupling cutoff must not be negative, got $(coupling_cutoff).")
    ewald_epsilon = args["ewald-epsilon"]
    0 < ewald_epsilon < 1 || error("The Ewald truncation error must lie in (0, 1), got $(ewald_epsilon).")
    ewald_eta = args["ewald-eta"]
    ewald_eta >= 0 || error("The Ewald splitting parameter must not be negative, got $(ewald_eta).")
    ewald_cost_ratio = args["ewald-cost-ratio"]
    ewald_cost_ratio > 0 || error("The Ewald cost ratio must be positive, got $(ewald_cost_ratio).")
    no_dipole_term = args["no-dipole-term"]
    band_map_plane = lowercase(args["band-map-plane"])
    band_map_points = args["band-map-points"]
    band_map_plane in ("none", "all", "xy", "xz", "yz") ||
        error("--band-map-plane must be none, all, xy, xz or yz, got '$(band_map_plane)'.")
    no_couplings = args["no-couplings"]
    transition_indices_str = args["transition-indices"]
    force_recomp = args["force-recomputation"]
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
    if crystal_mode && crystal_order == "coherent" && args["compute-rates"]
        error("Scattering rates are not supported for coherent crystal form factors because the rate calculation does not yet use the crystal band energies. Use --crystal-order incoherent or omit --compute-rates.")
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

            # Size the angular grid from l_max unless it was set explicitly.
            default_theta, default_phi = default_angular_grid(l_max)
            N_theta = N_theta > 0 ? N_theta : default_theta
            N_phi = N_phi > 0 ? N_phi : default_phi
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

                    # The coherent correction also needs ΔN and Ξ. They go through the same R
                    # contraction, but remain internal rather than changing the molecular output.
                    need_xi_and_dN = crystal_order == "coherent" && !no_couplings
                    xi_and_dN = nothing
                    if need_xi_and_dN
                        R_tensor, f_s, f_lm, transition_energies_eV, xi_and_dN =
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
                                need_flm=true
                            )
                    else
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
                    end

                    push!(conformer_f_lm, f_lm)
                    conformer_results[conformer_label] = (
                        R_tensor = R_tensor,
                        f_s = f_s,
                        f_lm = f_lm,
                        transition_energies_eV = transition_energies_eV,
                        xi_and_dN = xi_and_dN,
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

                # set_f_lm is aggregated over one conformer group (e.g. 1_A), the other is
                # aggregated over all groups. This f_lm-space path remains only for incoherent crystal
                # rates. The plotted incoherent form factor is built separately on the angular grid.
                need_incoherent_rates = compute_rates_flag
                set_f_lm, aggregate_f_lm = need_incoherent_rates ?
                    construct_crystal_f_lm_tensors(conformer_labels, conformer_f_lm, conformer_sets) :
                    (nothing, nothing)
                conformer_set_occupancies = T[conformer_set.occupancy for conformer_set in conformer_sets]

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

                    conformer_energy_lists = [T.(conformer_results[label].transition_energies_eV) for label in conformer_labels]
                    basis = build_excitation_basis(images, conformer_energy_lists)

                    # Rotate each conformer's coefficients into every image's orientation using D matrices.
                    # This keeps all images on the same unrotated q grid.
                    stage_times = Pair{String, Float64}[]
                    rotated_difference = nothing
                    rotated_Xi = nothing
                    if crystal_order == "coherent" && !no_couplings
                        conformer_difference = [conformer_results[label].xi_and_dN.difference_R
                                                for label in conformer_labels]
                        conformer_Xi = [conformer_results[label].xi_and_dN.Xi_R
                                        for label in conformer_labels]
                        rotate_timing = @timed rotate_crystal_R_tensors(
                            conformer_R_tensors, conformer_difference, conformer_Xi,
                            images, basis, l_max)
                        rotated_R, rotated_difference, rotated_Xi = rotate_timing.value
                    else
                        rotate_timing = @timed rotate_R_tensors(
                            conformer_R_tensors, images, basis, l_max)
                        rotated_R = rotate_timing.value
                    end
                    push!(stage_times, "rotate R tensors" => rotate_timing.time)

                    # The Brillouin zone folding works in inverse Angstroms, matching the lattice.
                    q_grid_invA = T(KEV_TO_INV_ANGSTROM) .* q_grid

                if crystal_order != "coherent"
                    # Nothing here depends on k, so there is no Bloch problem to solve: no couplings,
                    # no Ewald sum, no symmetry stars and no diagonalisation.
                    incoherent_timing = @timed compute_incoherent_crystal_f_lm(
                        rotated_R, basis, q_grid_invA, theta_grid, phi_grid, l_max;
                        need_grid = need_grid)
                    crystal_state_f_lm, crystal_f_s = incoherent_timing.value
                    push!(stage_times, "incoherent f_lm" => incoherent_timing.time)

                    print_stage_timings("Crystal stage timings", stage_times)

                    crystal_results = (
                        order = "incoherent",
                        basis = basis,
                        lattice = lattice,
                        dominant_group = dominant_group,
                        dominant_occupancy = dominant_occupancy,
                        state_f_lm = crystal_state_f_lm,
                        f_s = crystal_f_s,
                        band_summary = nothing,
                        couplings = nothing,
                        cells = nothing,
                        ewald_parameters = nothing,
                        long_range = nothing,
                        band_planes = String[],
                        band_maps = nothing,
                    )
                else

                    # Build the intermolecular couplings, unless they have been switched off.
                    crystal_couplings = nothing
                    ewald_parameters = nothing
                    crystal_long_range = nothing
                    cells = nothing
                    if no_couplings
                        println("Intermolecular corrections J and D are switched off, so the bands will be flat at the monomer energies.")
                    else
                        if coupling_method == "ewald"
                            ewald_parameters = choose_ewald_parameters(
                                lattice, ewald_epsilon; eta = ewald_eta, cost_ratio = ewald_cost_ratio)
                            # Pad the cutoff to not miss any molecules. Extra molecules will be dropped later.
                            real_cutoff = ewald_parameters.R_max + image_translation_span(basis)
                        else
                            real_cutoff = T(coupling_cutoff)
                        end

                        cells = enumerate_neighbour_cells(lattice, real_cutoff)

                        # Precompute the Gaunt coefficients for the coupling.
                        coupling_gaunt_path = joinpath(@__DIR__, "..", "src", "data", "gaunt_coefficients",
                                                       "gaunt_coefficients_coupling_lmax$(l_max)$(type_suffix).h5")
                        if force_recomp || !isfile(coupling_gaunt_path)
                            precompute_gaunt_coefficients(l_max, l_max, 2 * l_max, coupling_gaunt_path, T)
                        end

                        if ewald_parameters === nothing
                            println("Computing intermolecular corrections directly out to $(coupling_cutoff) Å ($(length(cells.vectors)) cells).")
                            correction_timing = @timed compute_crystal_corrections(
                                rotated_R, rotated_difference, rotated_Xi,
                                basis, cells, q_grid_invA, l_max, coupling_gaunt_path)
                            crystal_couplings, diagonal_corrections = correction_timing.value
                            set_diagonal_corrections!(basis, diagonal_corrections)
                            push!(stage_times, "couplings J and diagonal D" => correction_timing.time)
                        else
                            tau_of_lambda = [basis.images[basis.image_of[lambda]].translation
                                             for lambda in 1:length(basis.energies)]
                            long_range_timing = @timed build_ewald_long_range(
                                rotated_R, tau_of_lambda, basis.image_of, lattice, q_grid_invA, l_max,
                                ewald_parameters; include_dipole_term = !no_dipole_term)
                            crystal_long_range = long_range_timing.value
                            push!(stage_times, "Ewald long-range setup" => long_range_timing.time)

                            # Print a summary of the Ewald parameters.
                            expected_Q = lattice.volume * ewald_parameters.Q_max^3 / (6 * π^2)
                            supercell = supercell_radius(lattice)
                            widened = ewald_parameters.R_max <= supercell * (1 + 1e-12) ?
                                " (widened to the 3x3x3 supercell)" : ""
                            println("Ewald split at ε = $(ewald_parameters.epsilon): " *
                                    "η = $(round(ewald_parameters.eta, digits = 4)) Å^-1, " *
                                    "R_max = $(round(ewald_parameters.R_max, digits = 2)) Å$(widened) " *
                                    "($(length(cells.vectors)) cells searched, padded by the cell span), " *
                                    "Q_max = $(round(ewald_parameters.Q_max, digits = 4)) Å^-1 " *
                                    "(N_Q ≈ $(round(Int, expected_Q)), $(length(crystal_long_range.G_vectors)) candidates), " *
                                    "ℓ ceiling $(crystal_long_range.l_max_lr) of $(l_max).")
                            no_dipole_term && println("The Q = 0 term is dropped, so this is the conducting boundary condition.")

                            correction_timing = @timed compute_crystal_corrections(
                                rotated_R, rotated_difference, rotated_Xi,
                                basis, cells, q_grid_invA, l_max, coupling_gaunt_path;
                                parameters = ewald_parameters)
                            crystal_couplings, diagonal_short_range = correction_timing.value
                            push!(stage_times, "short-range J and D(ΔR)" => correction_timing.time)
                            subtract_self_term!(crystal_couplings, cells, crystal_long_range.self_term)

                            diagonal_timing = @timed compute_ewald_diagonal(
                                rotated_difference, rotated_Xi, q_grid_invA, l_max,
                                crystal_long_range)
                            set_diagonal_corrections!(
                                basis, diagonal_short_range .+ diagonal_timing.value)
                            push!(stage_times, "Ewald long-range D" => diagonal_timing.time)
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

                    # This streams over q internally, projecting each block onto real spherical
                    # harmonics as it goes, so the full |f|^2 grid is never held in memory. Overwritten
                    # need grid is specfied.
                    coherent_timing = @timed compute_coherent_crystal_f_lm(
                        rotated_R, basis, lattice, q_grid_invA, theta_grid, phi_grid, l_max,
                        stars;
                        need_grid = need_grid, couplings = crystal_couplings,
                        long_range = crystal_long_range)
                    crystal_state_f_lm, crystal_band_summary, crystal_f_s = coherent_timing.value
                    push!(stage_times, "coherent f_lm (H(k), diagonalise, project)" => coherent_timing.time)

                    # Optionally sample E_Ψ(k) over a plane of the first Brillouin zone, for plotting.
                    # Everything it needs is already built, so this is just the sweep, and it inherits
                    # the run's own q grid, ε and l_max rather than being told them a second time.
                    band_planes = band_map_plane == "none" ? String[] :
                                  band_map_plane == "all" ? ["xy", "xz", "yz"] : [band_map_plane]
                    band_maps = Dict{String, Tuple{Tuple{Vector{T}, Vector{T}}, Array{T, 3}}}()
                    if !isempty(band_planes)
                        band_timing = @timed for plane in band_planes
                            band_maps[plane] = sample_band_plane(
                                basis, lattice, crystal_couplings, crystal_long_range,
                                plane, band_map_points, T)
                        end
                        push!(stage_times,
                              "band map ($(join(band_planes, ", ")))" => band_timing.time)
                    end

                    print_stage_timings("Crystal stage timings", stage_times)
                    println()

                    crystal_results = (
                        order = "coherent",
                        basis = basis,
                        lattice = lattice,
                        dominant_group = dominant_group,
                        dominant_occupancy = dominant_occupancy,
                        band_summary = crystal_band_summary,
                        state_f_lm = crystal_state_f_lm,
                        f_s = crystal_f_s,
                        couplings = crystal_couplings,
                        cells = cells,
                        ewald_parameters = ewald_parameters,
                        long_range = crystal_long_range,
                        band_planes = band_planes,
                        band_maps = band_maps,
                    )
                end
                end


                # One file per crystal, either order. Coherent is indexed by crystal state Ψ, which
                # is a mixture of monomer transitions and so has no per-transition decomposition;
                # incoherent is indexed by monomer transition. The layout is otherwise the same, so
                # the plotting does not care which it is beyond reading crystal_order.
                if crystal_results !== nothing
                    crystal_output_dir = joinpath(mol_output_dir, "crystal")
                    mkpath(crystal_output_dir)

                    coherent_basis = crystal_results.basis

                    coherent_path = joinpath(crystal_output_dir, "crystal_f_lm$(type_suffix).h5")
                    h5open(coherent_path, "w") do io
                        # Which order (coherent or incoherent) produced this, so a reader knows what the first axis indexes.
                        write(io, "crystal_order", crystal_results.order)

                        # Coherent: crystal states Ψ, ordered by ascending energy at each q.
                        # Incoherent: monomer transitions.
                        write(io, "state_f_lm", crystal_results.state_f_lm)

                        # The unsquared form factor on the (q, θ, ϕ) grid, if it was asked for.
                        if crystal_results.f_s !== nothing
                            write(io, "f_s", crystal_results.f_s)
                            write(io, "theta_grid", theta_grid)
                            write(io, "phi_grid", phi_grid)
                        end
                        if crystal_results.band_summary !== nothing
                            write(io, "band_energy_min_eV", crystal_results.band_summary[:, 1])
                            write(io, "band_energy_max_eV", crystal_results.band_summary[:, 2])
                            write(io, "band_energy_mean_eV", crystal_results.band_summary[:, 3])
                        end

                        # Unperturbed monomer energies are present for both crystal orders. The
                        # environment shifts belong only to the coherent Frenkel Hamiltonian.
                        write(io, "localised_energies_eV", coherent_basis.energies)
                        if crystal_results.order == "coherent"
                            write(io, "diagonal_corrections_eV", coherent_basis.diagonal_corrections)
                        end

                        # Save the lattice vectors and image translations.
                        write(io, "lattice_A", Matrix{T}(crystal_results.lattice.direct))
                        write(io, "image_translations_A",
                              reduce(hcat, [image.translation for image in coherent_basis.images]))
                        write(io, "image_of", coherent_basis.image_of)

                        # Save everything needed to rebuild H(k), and hence E_Ψ(k) and C(k), at any k
                        # later:
                        #
                        #     H(k) = diag(E + D) + Σ_ΔR J^SR(ΔR) exp(i k . ΔR) + 𝒥^LR(k).
                        if crystal_results.couplings !== nothing
                            couplings_group = create_group(io, "couplings")
                            write(couplings_group, "method", coupling_method)
                            write(couplings_group, "cell_vectors", reduce(hcat, crystal_results.cells.vectors))
                            # The real-space sum over ΔR. Under "ewald" this is the short-range half
                            # alone, with the long-range self interaction already subtracted at ΔR = 0.
                            # Under "direct" it is the whole coupling, and there is no long_range group.
                            write(couplings_group, "J_real_space_eV", crystal_results.couplings.values)

                            if crystal_results.long_range !== nothing
                                long_range = crystal_results.long_range
                                # The reciprocal-space half, 𝒥^LR(k). It depends on k, so what is saved
                                # is the ingredients that rebuild it rather than a table over ΔR.
                                long_range_group = create_group(couplings_group, "long_range")
                                write(long_range_group, "eta", crystal_results.ewald_parameters.eta)
                                write(long_range_group, "epsilon", crystal_results.ewald_parameters.epsilon)
                                write(long_range_group, "R_max", crystal_results.ewald_parameters.R_max)
                                write(long_range_group, "Q_max", crystal_results.ewald_parameters.Q_max)
                                write(long_range_group, "G_vectors", reduce(hcat, long_range.G_vectors))
                                write(long_range_group, "f_table", long_range.f_table)
                                write(long_range_group, "q_step", long_range.q_step)
                                write(long_range_group, "l_max_lr", long_range.l_max_lr)
                                write(long_range_group, "dipole_slopes", long_range.dipole_slopes)
                                write(long_range_group, "include_dipole_term", long_range.include_dipole_term)
                            end
                        end

                        # The band energies over a plane of the first Brillouin zone, if asked for.
                        if !isempty(crystal_results.band_planes)
                            band_group = create_group(io, "band_map")
                            write(band_group, "planes", crystal_results.band_planes)
                            for plane in crystal_results.band_planes
                                (plane_axes, plane_energies) = crystal_results.band_maps[plane]
                                plane_group = create_group(band_group, plane)
                                write(plane_group, "energies_eV", plane_energies)
                                write(plane_group, "axis_a_keV", plane_axes[1])
                                write(plane_group, "axis_b_keV", plane_axes[2])
                            end
                        end

                        # What went in.
                        write(io, "transition_indices", collect(transition_indices))
                        write(io, "disorder_group", crystal_results.dominant_group)
                        write(io, "disorder_group_occupancy", crystal_results.dominant_occupancy)
                        write(io, "q_grid", q_grid)
                    end

                    println("$(titlecase(crystal_results.order)) crystal form factor saved to $(coherent_path).")
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
