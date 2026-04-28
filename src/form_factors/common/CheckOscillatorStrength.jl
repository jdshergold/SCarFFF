# This script checks the oscillator strength computed from the form factor against the PySCF value.

using HDF5
using Integrals
using ArgParse

const ELECTRON_MASS_KEV = 510.99895
const INV_FOUR_PI = 1 / (4*π)
const OSC_STRENGTH_THRESHOLD = 1e-5 # Threshold below which oscillator strengths are considered effectively zero.

@inline function compute_transition_dipole_moment(
        average_form_factor::Vector{T},
        q_subset::AbstractVector{T},
    ) where {T<:AbstractFloat}
    """
    Compute the transition dipole moment from the angular averaged form factor. This follows from the
    low-q expansion of the form factor:

        f(q) ∼ q⋅<f|r|i> + ... = q⋅μ + ...,

    with μ the transition dipole moment. The square of the form factor is therefore:

        |f(q)|^2 ∼ q^2⋅|μ|^2 + ...,

    and its angular average is:

        <|f(q)|^2>_Ω ∼ q^2 |μ|^2 / 3 + ....

    This tells us that:

        lim_{q→0} <|f(q)|^2>_Ω / q^2 = |μ|^2 / 3,

    which to be well behaved requires that <|f(q)|^2> and d<|f(q)|^2>/dq vanish at q=0. Thus we can compute |μ|^2 as:

        (2/3) * |μ|^2 = (d/dq)^2 <|f(q)|^2>_Ω (at q = 0),

    where the factor of 2 pops out from the Taylor expansion. Since we have to evaluate the derivative on
    a grid at points q > 0, also keep the q^4 terms and perform a fit. That is, we write:

        <|f(q)|^2>_Ω = A q^2 + B q^4 + ...,

    and find A = |μ|^2 / 3 by fitting. We also include a constant term in the fit, so that we can
    subtract any non-zero form factor at q=0 due to numerical noise.

    # Arguments:
    - average_form_factor::Vector{T}: The angular averaged form factor values.
    - q_subset::Vector{T}: The subset of |q| values, in keV, up to q_max_fit.

    # Returns:
    - T: The squared transition dipole moment, (1/3) * |μ|^2 in units of keV^{-2}.
    """

    # Define the model to fit. For us, 1 + q^2 + q^4. This is a matrix of shape (N, 3) where N is the number of q-points.
    n = length(q_subset)
    A = Matrix{T}(undef, n, 3)
    @inbounds for i in 1:n
        q = q_subset[i]
        A[i, 1] = one(T)
        A[i, 2] = q * q
        A[i, 3] = q * q * q * q
    end

    # Perform the linear least squares fit. For square matrices A \ B in Julia solves Ax = B for x.
    # Here, x would be our coefficients, and B our average form factor.
    # In our case, A is not a square matrix, so there is no inverse. Instead, Julia tries to minimise ||Ax - B||.
    # All of this magic is in one command!

    coeffs = A \ average_form_factor

    return coeffs[2] # This is |μ|^2 / 3.
end

@inline function perform_angular_average(f_s_sq::Array{T, 3}, theta_grid::Vector{T}, phi_grid::Vector{T}) where {T<:AbstractFloat}
    """
    Perform the angular average of the squared form factor |f_s(q, θ, ϕ)|^2 over the spherical angles θ and ϕ:

        <|f_s(q)|^2>_Ω = (1/4π) ∫_0^{2π} dϕ ∫_0^π dθ sin(θ) |f_s|^2.

    We do this in place to avoid allocating the output array.

    # Arguments:
    - f_s_sq::Array{T, 3}: The squared form factor on the (q, θ, ϕ) grid.
    - theta_grid::Vector{T}: The θ grid.
    - phi_grid::Vector{T}: The ϕ grid.

    # Returns:
    - Vector{T}: The angular averaged form factor.
    """

    # Allocate the intermediate arrays.
    n_q = size(f_s_sq, 1)
    n_theta = size(f_s_sq, 2)

    f_s_phi = Array{T, 2}(undef, n_q, n_theta)

    # Now perform the phi average using Simpson's rule.
    method = SimpsonsRule()
    phi_problem = SampledIntegralProblem(f_s_sq, phi_grid; dim = 3)
    f_s_phi .= solve(phi_problem, method).u

    # Do the same for the theta average, including the Jacobian.
    @inbounds for j in 1:n_theta
        sin_theta_j = sin(theta_grid[j])
        @inbounds for i in 1:n_q
            f_s_phi[i, j] *= sin_theta_j
        end
    end

    theta_problem = SampledIntegralProblem(f_s_phi, theta_grid; dim = 2)

    return solve(theta_problem, method).u .* T(INV_FOUR_PI)
end

function compute_cartesian_radial_average(
        form_factor::Array{Complex{T}, 3},
        qx_grid::Vector{T},
        qy_grid::Vector{T},
        qz_grid::Vector{T},
        q_max_fit::T,
    ) where {T<:AbstractFloat}
    """
    Compute the angular averaged |f(q)|^2 on a Cartesian q-grid up to q_max_fit by binning in |q|.
    The method is roughly as follows:

        1) Trim the q-grids to only include points within ±q_max_fit.
        2) Determine the bin width from the smallest non-zero step in the q-grids.
        3) Loop over all trimmed (qx, qy, qz) points, computing |q| and binning |f(q)|^2 accordingly.
        4) Compute the average |f(q)|^2 and |q| in each bin.
        5) This gives us (<|q|>, <|f(q)|^2>) that can be used for fitting later.

    # Arguments:
    - form_factor::Array{S, 3}: The form factor on the (qx, qy, qz) grid.
    - qx_grid::Vector{T}: The qx grid.
    - qy_grid::Vector{T}: The qy grid.
    - qz_grid::Vector{T}: The qz grid.
    - q_max_fit::T: Maximum q value in keV for fitting.

    # Returns:
    - f_s_sq_av::Vector{T}: The binned and averaged |f(q)|^2 values.
    - q_av::Vector{T}: The binned and averaged |q| values.
    """

    max_q = q_max_fit

    # Trim the grids to avoid iterating over points that will be discarded.
    function trim_uniform_grid(grid)
        n = length(grid)
        if n < 2
            error("Unable to determine q-grid spacing for the oscillator strength computation.")
        end
        step = abs(grid[2] - grid[1])
        # Find the start and indices in a clever way, by starting at the limits and finding how many steps take us to q_max_fit.
        start_idx = clamp(Int(ceil(((-max_q - grid[1]) / step) + 1)), 1, n)
        end_idx = clamp(Int(floor(((max_q - grid[1]) / step) + 1)), 1, n)
        return start_idx, end_idx, step
    end

    # Trim each grid to the range [-Q_MAX, Q_MAX].
    qx_start, qx_end, qx_step = trim_uniform_grid(qx_grid)
    qy_start, qy_end, qy_step = trim_uniform_grid(qy_grid)
    qz_start, qz_end, qz_step = trim_uniform_grid(qz_grid)

    @views qx_trim = qx_grid[qx_start:qx_end]
    @views qy_trim = qy_grid[qy_start:qy_end]
    @views qz_trim = qz_grid[qz_start:qz_end]

    if isempty(qx_trim) || isempty(qy_trim) || isempty(qz_trim)
        error("Unable to find q-grid points within ±$(q_max_fit) keV for oscillator strength computation.")
    end

    # Determine the smallest non-zero step across all q-grids to set the bin width.
    q_steps = T[abs(qx_step), abs(qy_step), abs(qz_step)]

    if isempty(q_steps)
        error("Unable to determine q-grid spacing for oscillator strength computation.")
    end

    # The q window for binning. Any q within, e.g. [0, q_bin) goes into the first bin, [q_bin, 2*q_bin) into the second, etc.
    q_bin = minimum(q_steps)
    n_bins = max(1, Int(floor(max_q / q_bin)) + 1)

    # Allocate accumulators for binning.
    f_s_sq_sum = zeros(T, n_bins)
    q_sum = zeros(T, n_bins)
    counts = zeros(Int, n_bins)

    n_z = length(qz_trim)
    n_y = length(qy_trim)
    n_x = length(qx_trim)

    @inbounds for k in 1:n_z
        qz = qz_trim[k]
        for j in 1:n_y
            qy = qy_trim[j]
            for i in 1:n_x
                qx = qx_trim[i]
                q = sqrt(qx * qx + qy * qy + qz * qz)
                if q <= max_q
                    # Find the appropriate bin.
                    bin_idx = Int(floor(q / q_bin)) + 1
                    counts[bin_idx] += 1
                    # Accumulate the form factor squared and q value.
                    q_sum[bin_idx] += q
                    f_s_sq_sum[bin_idx] += abs2(form_factor[i, j, k])
                end
            end
        end
    end

    # Now we compute the averages for the non-zero bins, which will give us (q, |f(q)|^2) that can be fitted to later.
    non_zero_bins = findall(!=(0), counts)
    f_s_sq_av = Vector{T}(undef, length(non_zero_bins))
    q_av = Vector{T}(undef, length(non_zero_bins))

    @inbounds for (idx, bin_idx) in enumerate(non_zero_bins)
        inv_count = inv(T(counts[bin_idx]))
        f_s_sq_av[idx] = f_s_sq_sum[bin_idx] * inv_count
        q_av[idx] = q_sum[bin_idx] * inv_count
    end

    return f_s_sq_av, q_av
end

function compute_oscillator_strength(h5_file_path::String, q_max_fit::Real; method::String = "spherical")
    """
    Compute the oscillator strength from the tabulated form factor stored in an HDF5 file. This is defined
    in terms of the transition dipole moment μ as:

        f_osc = (2/3) * m_e * ΔE_{fi} * |μ|^2,

    in natural units, with ħ = c = 1, where ΔE_{fi} is the transition energy, and m_e is the electron mass.

    # Arguments:
    - h5_file_path::String: The path to the HDF5 file containing the form factor data.
    - q_max_fit::Real: Maximum q value in keV for fitting.

    # Returns:
    - Real: The oscillator strength.
    """

    # Load the form factor and momentum grid from the HDF5 file.
    method = lowercase(method)

    transition_energy_eV = zero(Float64)
    T = Float64
    f_s_sq_av = Float64[]
    q_subset = Float64[]

    if method == "spherical"
        f_s, q_grid, theta_grid, phi_grid, transition_energy_eV = h5open(h5_file_path, "r") do io
            (
                read(io, "f_s"),
                read(io, "q_grid"),
                read(io, "theta_grid"),
                read(io, "phi_grid"),
                read(io, "transition_energy_eV")
            )
        end

        # Find the indices up to q_max_fit, above which we would expect the fit to break down.
        T = eltype(q_grid)
        n_q = searchsortedlast(q_grid, T(q_max_fit))

        # Check that there are at least 2 q-points below q_max_fit for the fit.
        if n_q < 2
            error("Insufficient q-points in the form factor data to compute the oscillator strength. We need at least 2 q-points below $(q_max_fit) keV.")
        end

        # Compute the angular averaged form factor squared for the first q-points up to q_max_fit using Simpson's rule.
        # First allocate an array to hold |f_s|^2 for the subset of q values.
        n_theta = size(f_s, 2)
        n_phi = size(f_s, 3)
        f_s_sq_subset = Array{T, 3}(undef, n_q, n_theta, n_phi)
        @views q_subset = q_grid[1:n_q]

        # Now populate the array.
        @inbounds for q_idx in 1:n_q
            @inbounds for theta_idx in 1:n_theta
                @inbounds for phi_idx in 1:n_phi
                    f_s_sq_subset[q_idx, theta_idx, phi_idx] = abs2(f_s[q_idx, theta_idx, phi_idx])
                end
            end
        end

        # Perform the angular average.
        f_s_sq_av = perform_angular_average(f_s_sq_subset, theta_grid, phi_grid)

    elseif method == "fft" || method == "cartesian"
        form_factor, qx_grid, qy_grid, qz_grid, transition_energy_eV = h5open(h5_file_path, "r") do io
            (
                read(io, "form_factor"),
                read(io, "qx_grid"),
                read(io, "qy_grid"),
                read(io, "qz_grid"),
                read(io, "transition_energy_eV")
            )
        end

        T = eltype(qx_grid)
        f_s_sq_av, q_subset = compute_cartesian_radial_average(form_factor, qx_grid, qy_grid, qz_grid, T(q_max_fit))

    else
        error("Invalid method '$(method)'. Supported methods are 'spherical', 'fft', and 'cartesian'.")
    end

    # Check that there are at least 2 q-points below q_max_fit for the fit.
    if length(q_subset) < 2
        error("Insufficient q-points in the form factor data to compute the oscillator strength. We need at least 2 q-points below $(q_max_fit) keV.")
    end

    # Compute the squared transition dipole moment.
    mu_sq_o3 = compute_transition_dipole_moment(f_s_sq_av, q_subset) # Includes the 1/3 factor.

    # Construct the oscillator strength from what we have.
    transition_energy_keV = T(transition_energy_eV) * T(0.001)
    f_osc = T(2.0) * T(ELECTRON_MASS_KEV) * transition_energy_keV * mu_sq_o3
    return f_osc
end

function parse_commandline()::Dict{String, Any}
    """
    Parse the command line arguments.

    # Returns:
    - args::Dict{String, Any}: The parsed command line arguments.
    """
    s = ArgParseSettings(
        description = "Check the oscillator strength computed using the form factor against that from PySCF."
    )

    @add_arg_table! s begin
        "--smiles"
            help = "SMILES string for the molecule."
            required = true
        "--precision"
            help = "Precision to check the oscillator strength for. Options: float32 or float64."
            default = "float64"
        "--method"
            help = "Form factor computation method. Options: spherical, fft, or cartesian."
            default = "spherical"
        "--transition-indices"
            help = "Transition indices to verify. Can be 'all' or comma-separated list like '1,2,3'."
            default = "all"
        "--run-name"
            help = "Name of the run directory (defaults to SMILES if not provided)."
            default = nothing
        "--molecule-number"
            help = "Molecule number to verify (for batch runs)."
            arg_type = Int
            default = 1
        "--q-max-fit"
            help = "Maximum q value in keV for fitting the oscillator strength."
            arg_type = Float64
            default = 0.2
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    smiles = args["smiles"]
    precision = lowercase(args["precision"])
    method = lowercase(args["method"])
    transition_indices_str = args["transition-indices"]
    run_name = args["run-name"]
    molecule_number = args["molecule-number"]
    q_max_fit = args["q-max-fit"]

    molecule_number < 1 && error("Molecule number must be ≥ 1.")
    q_max_fit <= 0.0 && error("q_max_fit must be positive.")

    # Determine the type suffix based on the specified precision.
    type_suffix = precision == "float32" ? "_f32" : "_f64"

    # Use the run_name if provided, otherwise fall back to SMILES.
    dir_name = isnothing(run_name) ? smiles : run_name

    # Construct the paths to the relevant files, including molecule number and method subdirectories.
    runs_dir = joinpath(@__DIR__, "..", "..", "..", "runs", dir_name)
    mol_dir = joinpath(runs_dir, string(molecule_number))
    method_dir = joinpath(mol_dir, method)

    # Check that the method directory exists.
    if !isdir(method_dir)
        error("Method directory not found at $(method_dir). Please run compute_form_factor.jl first.")
    end

    # Determine which transitions to check.
    transitions_to_check = Int[]
    if lowercase(transition_indices_str) == "all"
        # Find all of the transition directories.
        for entry in readdir(method_dir)
            if startswith(entry, "transition_")
                trans_num_str = replace(entry, "transition_" => "")
                trans_num = tryparse(Int, trans_num_str)
                if !isnothing(trans_num)
                    push!(transitions_to_check, trans_num)
                end
            end
        end
        sort!(transitions_to_check)
    else
        # Parse the comma-separated list.
        for trans_str in split(transition_indices_str, ",")
            trans_num = parse(Int, strip(trans_str))
            trans_num < 1 && error("Transition index must be ≥ 1.")
            push!(transitions_to_check, trans_num)
        end
    end

    if isempty(transitions_to_check)
        error("No transitions found to verify.")
    end

    td_dft_file = joinpath(mol_dir, "td_dft_results$(type_suffix).h5")

    # Check that the TD-DFT file exists.
    if !isfile(td_dft_file)
        error("TD-DFT file not found at $(td_dft_file). Please run td_dft.py first with precision=$(precision).")
    end

    println("Oscillator strength verification:")
    println("  Method: $(method)")
    println("  Precision: $(precision)")
    println("  q_max_fit: $(q_max_fit) keV")
    println("  Transitions: $(join(transitions_to_check, ", "))")
    println()

    # Loop through each transition and verify.
    for requested_transition in transitions_to_check
        transition_dir = joinpath(method_dir, "transition_$(requested_transition)")
        form_factor_file = joinpath(transition_dir, "fs_grid$(type_suffix).h5")

        # Check that the form factor file exists.
        if !isfile(form_factor_file)
            println("⚠ Transition $(requested_transition): Form factor file not found, skipping.")
            println()
            continue
        end

        # Compute the oscillator strength from the form factor.
        f_osc_computed = compute_oscillator_strength(form_factor_file, q_max_fit, method=method)

        # Read the PySCF oscillator strength and transition index.
        transition_index = h5open(form_factor_file, "r") do io
            read(io, "transition_index")
        end

        f_osc_pyscf = h5open(td_dft_file, "r") do io
            read(io, "f_osc")[transition_index]
        end

        # Apply the oscillator strength threshold. If both values are below the threshold then we treat them as zero.
        if abs(f_osc_computed) < OSC_STRENGTH_THRESHOLD && abs(f_osc_pyscf) < OSC_STRENGTH_THRESHOLD
            f_osc_computed = 0.0
            f_osc_pyscf = 0.0
        end

        # Compute the relative difference.
        rel_diff = f_osc_pyscf == 0.0 ? 0.0 : abs(f_osc_computed - f_osc_pyscf) / f_osc_pyscf

        # Print results.
        println("Transition $(requested_transition):")
        println("  Computed value:        $f_osc_computed")
        println("  PySCF value:           $f_osc_pyscf")
        println("  Relative difference:   $(round(rel_diff * 100, digits=4))%")
        println()
    end

end

# Run the main function.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
