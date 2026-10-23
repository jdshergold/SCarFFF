"""Compute the form factor stage of a resolved YAML job."""

using SCarFFF
using HDF5, JSON, CUDA

const SF = SCarFFF.SphericalFormFactor
include("Results.jl")
using .MolecularResults: write_results, write_quantity
include("Verification.jl")

function spherical_grid(settings, ::Type{T}) where {T<:AbstractFloat}
    """
    Construct the spherical momentum grid.

    # Arguments:
    - settings::Dict: The spherical calculation settings.
    - T::Type: The floating point type.

    # Returns:
    - tuple: The q, theta and phi coordinates.
    """
    # |f|^2 is quadratic in the form factor, so it carries content up to 2 l_max, needing 4 l_max + 1 points.
    default_count = max(3, 4 * settings["l_max"] + 1)
    n_theta = settings["n_theta"] == 0 ? default_count : settings["n_theta"]
    n_phi = settings["n_phi"] == 0 ? default_count : settings["n_phi"]
    return collect(range(T(0), T(settings["q_max"]); length=settings["n_q"])),
           collect(range(T(0), T(π); length=n_theta)),
           collect(range(T(0), T(2π); length=n_phi))
end

function verify_molecular_result(path, td_path, q_max_fit)
    """
    Compare fitted oscillator strengths with TD-DFT for each saved transition.

    # Arguments:
    - path::String: The grouped molecular form factors.
    - td_path::String: The TD-DFT results.
    - q_max_fit::Real: The maximum fitted momentum, in keV.
    """
    # The oscillator strengths TD-DFT reported.
    reference = h5open(io -> read(io, "f_osc"), td_path, "r")
    h5open(path, "r") do io
        grid = io["grid"]
        for name in keys(io["transitions"])
            state = io["transitions/$name"]
            haskey(state, "f_s") || error("The oscillator strength check requires saved f_s.")
            amplitude = read(state, "f_s")
            if read(HDF5.attributes(io)["mode"]) == "spherical"
                # Average |f|^2 over the angles, then read the dipole off its small q behaviour.
                q = read(grid, "q")
                n_q = searchsortedlast(q, q_max_fit)
                n_q >= 2 || error("The oscillator strength check needs two q points below q_max_fit.")
                mean = perform_angular_average(abs2.(amplitude[1:n_q, :, :]), read(grid, "theta"), read(grid, "phi"))
                moment = compute_transition_dipole_moment(mean, q[1:n_q])
            else
                # The Cartesian grids have no angular axes, so bin |f|^2 by |q| instead.
                q_x, q_y, q_z = read(grid, "q_x"), read(grid, "q_y"), read(grid, "q_z")
                mean, q = compute_cartesian_radial_average(amplitude, q_x, q_y, q_z, eltype(q_x)(q_max_fit))
                length(q) >= 2 || error("The oscillator strength check needs two radial samples.")
                moment = compute_transition_dipole_moment(mean, q)
            end

            # f_osc = 2 m_e dE |mu|^2, with the 0.001 taking dE from eV to keV.
            fitted = 2 * ELECTRON_MASS_KEV * read(state, "Delta_E") * 0.001 * moment
            println("Transition $name: fitted oscillator strength $fitted; TD-DFT $(reference[parse(Int, name)]).")
        end
    end
end

function compute_molecule(job, system)
    """
    Compute and save all requested transitions for one molecule.

    # Arguments:
    - job::Dict: The resolved job settings.
    - system::Dict: The numbered molecule and calculation signature.
    """
    # Unpack the job settings.
    T = job["execution"]["precision"] == "float32" ? Float32 : Float64
    use_gpu = job["execution"]["gpu"]
    use_gpu && !CUDA.functional() && error("A GPU was requested but CUDA is not functional.")
    directory = joinpath(job["output"], string(system["number"]))
    td_path = joinpath(directory, "td_dft.h5")
    indices = job["transitions"] == "all" ? collect(1:job["nstates"]) : Int.(job["transitions"])
    mode = job["mode"]

    # Only the quantities the chosen method produces are filled in below.
    grid = Dict{String, Any}()
    f_s = f_sq_lm = R_tensor = density = V = nothing
    if mode == "spherical"
        settings = job["spherical"]
        q, theta, phi = spherical_grid(settings, T)
        save = settings["save"]
        R_tensor, f_s, f_sq_lm, energies = compute_spherical_form_factor(
            q, theta, phi, settings["l_max"], td_path;
            transition_indices=indices, force_recomputation=settings["force_recomputation"],
            threshold=T(settings["threshold"]), use_gpu=use_gpu,
            need_grid="f_s" in save, need_R="R_tensor" in save,
            need_f_sq_lm="f_sq_lm" in save)
        grid["q"] = (q, "keV")
        grid["theta"] = (theta, "rad")
        grid["phi"] = (phi, "rad")
    else
        settings = job[mode]
        coordinates = Vector{T}[]
        if mode == "fourier"
            # Build a symmetric q grid along each axis from its limit and spacing.
            for (limit, spacing) in zip(settings["q_max"], settings["dq"])
                count = round(Int, 2 * limit / spacing) + 1
                push!(coordinates, collect(range(T(-limit), T(limit); length=count)))
            end
            f_s, density, r_lim, energies = compute_fft_form_factor(
                coordinates..., td_path; transition_indices=indices,
                check_parseval=settings["check_parseval"], use_gpu=use_gpu)
            # The FFT returns its own q ordering, and a real space grid to go with the density.
            for i in 1:3
                n = length(coordinates[i])
                spacing = coordinates[i][2] - coordinates[i][1]
                coordinates[i] = T.(collect(-fld(n, 2):(cld(n, 2)-1))) .* spacing
                grid[("x", "y", "z")[i]] = (T.([-r_lim[i] + j*2*r_lim[i]/n for j in 0:(n-1)]), "angstrom")
            end
        else
            # The Cartesian grids are given directly as a range per axis.
            for name in ("q_x", "q_y", "q_z")
                low, high, count = settings[name]
                push!(coordinates, collect(range(T(low), T(high); length=Int(count))))
            end
            V, f_s, energies = compute_cartesian_form_factor(coordinates..., td_path;
                transition_indices=indices, threshold=T(settings["threshold"]), use_gpu=use_gpu,
                need_grid="f_s" in settings["save"], need_V="V" in settings["save"])
        end
        for (name, values) in zip(("q_x", "q_y", "q_z"), coordinates)
            grid[name] = (values, "keV")
        end
    end

    # Save everything, optionally checking the file against TD-DFT before it goes into place.
    path = joinpath(directory, "form_factors.h5")
    check_result = job["verification"]["oscillator_strength"] ?
        temporary -> verify_molecular_result(temporary, td_path, job["verification"]["q_max_fit"]) : nothing
    write_results(path, mode, system["form_factor_signature"], grid, indices, energies;
                  f_s=f_s, f_sq_lm=f_sq_lm, R_tensor=R_tensor, transition_density=density,
                  V_tensors=V, check_result=check_result)
end

function run_form_factors(job)
    """
    Run the requested molecular or crystal form factor calculations.

    # Arguments:
    - job::Dict: The resolved job received from Python.
    """
    if job["mode"] == "crystal"
        # Load the crystal path only when it is wanted.
        include("CrystalJob.jl")
        failures = Base.invokelatest() do
            run_crystals(job)
        end
    else
        # One molecule at a time, carrying on after a failure so the rest still get computed.
        failures = String[]
        for system in job["systems"]
            try
                compute_molecule(job, system)
            catch error
                @error "Molecular form factor calculation failed" molecule=system["number"] exception=(error, catch_backtrace())
                push!(failures, string(system["number"]))
            end
        end
    end
    isempty(failures) || error("Form factor calculations failed for: $(join(failures, ", ")).")
end

# The job arrives as JSON on stdin when Python starts this file.
if abspath(PROGRAM_FILE) == @__FILE__
    run_form_factors(JSON.parse(read(stdin, String)))
end
