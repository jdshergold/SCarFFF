# This module contains functions to compute the DM scattering rate using VSDM.

module ComputeRates

using Base.Threads
using VectorSpaceDarkMatter
using Quaternionic
import Logging: NullLogger, with_logger

include("../../utils/DMModels.jl")
using .DMModels: spherical_halo, v_max

const VSDM = VectorSpaceDarkMatter

function to_tophat(flm::Array{T, 3}) where {T<:AbstractFloat}
    """
    Convert the FLM tensor to a tophat harmonic basis compatible with VSDM. Explicitly, we define:

        f^2_{nℓm} = √(3/(x_{n+1}^3 - x_{n}^3)) ∫_{x_n}^{x_{n+1}} x^2 dx f^2_{ℓm}(x*qmax)
                  ≃ √(x_{n+1}^3 - x_{n}^3)/3) F^2_{nℓm}  

    where x_n = q/qmax = n/N_q, and:

        F^2_{nℓm} = 1/2 * (f^2_{ℓm}(x_n*qmax) + f^2_{ℓm}(x_{n+1}*qmax)).

    This takes N_q+1 flm coefficients and turns them into N_q tophat coefficients.

    # Arguments:
    - flm::Array{T, 3}: The f^2_{ℓm} tensor with dimensions (n_transitions, n_q, n_keys), keyed by key(ℓ, m) = ℓ^2 + (ℓ + m) + 1.

    # Returns:
    - tophat::Array{T, 3}: The tophat tensor with dimensions (n_transitions, n_q-1, n_keys), keyed by key(ℓ, m) = ℓ^2 + (ℓ + m) + 1.
    """

    # Get the dimensions.
    N_transitions = size(flm, 1)
    N_q = size(flm, 2)
    N_lm = size(flm, 3)
    n_bins = N_q - 1
    inv_n_bins = inv(T(n_bins))

    # Preallocate the tophat tensor.
    tophat = zeros(T, N_transitions, n_bins, N_lm)

    # Precompute the weights for each bin.
    weights = Vector{T}(undef, n_bins)
    @inbounds for q_idx in 1:n_bins
        x_n = (q_idx - 1) * inv_n_bins
        x_np1 = q_idx * inv_n_bins
        weights[q_idx] = sqrt((x_np1^3 - x_n^3) / 3)
    end

    # Compute the tophat coefficients.
    @threads for lm_idx in 1:N_lm
        @inbounds for q_idx in 1:n_bins
            weight = weights[q_idx]
            for n in 1:N_transitions
                F2 = 0.5 * (flm[n, q_idx, lm_idx] + flm[n, q_idx + 1, lm_idx])
                tophat[n, q_idx, lm_idx] = weight * F2
            end
        end
    end

    return tophat
end

function compute_rates(
        flm::Array{T,3},
        q_grid::Vector{T},
        m_grid::Vector{T},
        transition_energies::Vector{T},
        N_rotations::Union{Nothing,Tuple{Int,Int,Int}}=nothing,
        model=spherical_halo(),
    ) where {T<:AbstractFloat}
    """
    Compute the DM scattering rate using VSDM for a set of detector orientations. Currently
    uses one tophat wavelet per q-value, and the same number of velocity values.

    # Arguments:
    - flm::Array{T, 3}: The f^2_{ℓm} tensor with dimensions (n_transitions, n_q, n_keys), keyed by key(ℓ, m) = ℓ^2 + (ℓ + m) + 1.
    - q_grid::Vector{T}: The momentum transfer grid in keV.
    - m_grid::Vector{T}: The DM mass grid in MeV.
    - transition_energies::Vector{T}: The transition energies in eV.
    - N_rotations::Tuple{Int,Int,Int}: The number of (α,β,γ) detector rotations to consider. If unset, considers only the identity.
    - model::Function: The DM velocity model function (default: spherical_halo).

    # Returns:
    - results::Vector{NamedTuple{(:mchi_MeV, :rate_max, :rate_min, :rate_mean), NTuple{4,T}}}: The dimensionless rates for each DM mass, after summing over transitions and taking the max/min/mean over orientations.
    """

    # Get the stuff needed to setup VSDM.
    N_transitions = size(flm, 1)
    l_max = Int(sqrt(size(flm, 3)) - 1) # N_lm = (lmax + 1)^2.
    q_max = q_grid[end] # In keV.

    # First compute the tophat coefficients for the f^2_{ℓm} tensor.
    f_tophat = to_tophat(flm)
    n_v = size(f_tophat, 2)

    # Setup the VSDM ingredients.
    xv_grid = collect(range(0.0,1.0,n_v+1)) # Normalised velocity grid.
    xq_grid = Float64.(q_grid ./ q_max) # Normalised momentum grid.
    q_basis = VSDM.Tophat(xq_grid, Float64(q_max) * VSDM.keV)
    v_basis = VSDM.Tophat(xv_grid, v_max * VSDM.km_s)

    # Setup the rotation grid.
    if N_rotations === nothing
        R_arr = vec([from_euler_angles(0.0, 0.0, 0.0)])
    else
        # Generate the rotation grid. We create N+1 rotations and drop the last one for periodicity.
        α_vals = range(0.0,2π,N_rotations[1]+1)[1:end-1]
        β_vals = range(0.0,π,N_rotations[2]+1)[1:end-1]
        γ_vals = range(0.0,2π,N_rotations[3]+1)[1:end-1]
        R_arr = vec([from_euler_angles(α, β, γ) for α in α_vals, β in β_vals, γ in γ_vals])
    end

    # Construct the projected form factor objects that VSDM expects.
    lm = VSDM.LM_vals(l_max)
    fs_list = Vector{VSDM.ProjectedF{Float64, typeof(q_basis)}}(undef, N_transitions)
    for transition_idx in 1:N_transitions
        fs_list[transition_idx] = VSDM.ProjectedF(Float64.(@view f_tophat[transition_idx, :, :]), lm, q_basis)
    end

    # Project the velocity distribution onto the tophat basis.
    # Here n_v is the number of tophat bins, so VSDM wants n_max = n_v - 1.
    v_tophat = VSDM.ProjectF(model,(n_v-1,l_max),v_basis)

    # Compute dimensionless the scattering rate here using VSDM.
    # The orientation is a single physical crystal orientation, so rates from
    # all transitions are summed at each R before taking the max.
    mSM = VSDM.mElec
    N_rotations = length(R_arr)
    inv_N_rotations = inv(Float64(N_rotations))
    transition_energies_eV = Float64.(transition_energies)
    null_logger = NullLogger()
    results = Vector{NamedTuple{(:mchi_MeV, :rate_max, :rate_min, :rate_mean), NTuple{4,T}}}(undef, length(m_grid))

    # Sweep over DM masses. For each mass, sum rates over all transitions at
    # each orientation, then record the max/min/mean over orientations.
    @threads for i in eachindex(m_grid)
        mchi = Float64(m_grid[i]) * VSDM.MeV
        total_rate = zeros(Float64, N_rotations)

        # Use the logger to suppress SpecialFunctions warnings.
        with_logger(null_logger) do
            for transition_idx in 1:N_transitions
                dm_model = VSDM.ModelDMSM(0, mchi, mSM, transition_energies_eV[transition_idx] * VSDM.eV)
                total_rate .+= VSDM.rate(R_arr, dm_model, v_tophat, fs_list[transition_idx])
            end
        end

        # Store the rates.
        rate_max = -Inf
        rate_min = Inf
        rate_sum = 0.0
        @inbounds for rate in total_rate
            rate_max = max(rate_max, rate)
            rate_min = min(rate_min, rate)
            rate_sum += rate
        end

        results[i] = (
            mchi_MeV = m_grid[i],
            rate_max = T(rate_max),
            rate_min = T(rate_min),
            rate_mean = T(rate_sum * inv_N_rotations),
        )
    end

    return results
end


end

"""
# Testing.

if abspath(PROGRAM_FILE) == @__FILE__
    using HDF5
    using Quaternionic
    using VectorSpaceDarkMatter
    import Logging: with_logger, NullLogger

    const VSDM = VectorSpaceDarkMatter

    # Collect all transition directories under runs/flmsq_test/1/spherical/.
    project_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    default_spherical_dir = joinpath(project_root, "runs", "flmsq_test", "1", "spherical")
    spherical_dir = isempty(ARGS) ? default_spherical_dir : ARGS[1]

    transition_paths = sort(filter(
        p -> isfile(joinpath(p, "fs_grid_f32.h5")),
        readdir(spherical_dir, join=true),
    ))
    isempty(transition_paths) && error("No transition directories found in spherical_dir")
    println("Found (length(transition_paths)) transitions.")

    # Load all transition form factors and convert to tophat.
    pfq_list = ProjectedF[]
    transition_energies = Float64[]

    for tpath in transition_paths
        f_lm, q_grid, dE = h5open(joinpath(tpath, "fs_grid_f32.h5"), "r") do file
            read(file["f_lm"]), read(file["q_grid"]), read(file["transition_energy_eV"])
        end
        f_lm_tensor = ndims(f_lm) == 2 ? reshape(f_lm, 1, size(f_lm, 1), size(f_lm, 2)) : f_lm
        tophat = ComputeRates.to_tophat(f_lm_tensor)

        # Drop transition index for now.
        tophat = reshape(tophat, size(tophat, 2), size(tophat, 3))

        q_max = q_grid[end] * VSDM.keV # In keV.
        local lmax = sqrt(size(f_lm_tensor, 3)) - 1
        x_grid = q_grid ./ q_grid[end]

        q_basis = VSDM.Tophat(x_grid, q_max)
        lm = VSDM.LM_vals(Int(lmax))

        push!(pfq_list, VSDM.ProjectedF(Float64.(tophat), lm, q_basis))
        push!(transition_energies, Float64(dE))
    end

    mSM = VSDM.mElec

    # Spherical halo model definition here:

    v0 = 220.0 * VSDM.km_s
    ve = 230.0 * VSDM.km_s
    v_max = 960.0 * VSDM.km_s

    function shm(v0, ve)
        return VSDM.GaussianF(1.0, [ve, 0.0, 0.0], v0 / sqrt(2))
    end

    # Get the gchi tophat functions here using VSDM.
    n_v = size(pfq_list[1].fnlm, 1)
    lmax = maximum(lm[1] for pf in pfq_list for lm in pf.lm)
    v_grid = collect(range(0.0, 1.0, n_v + 1))
    v_basis = VSDM.Tophat(v_grid, v_max)
    pfv = VSDM.ProjectF(shm(v0, ve), (n_v - 1, lmax), v_basis)

    # Compute dimensionless scattering rate here using VSDM.
    # Sweep over orientations using ZYZ Euler angles (α, β, γ).
    # The orientation is a single physical crystal orientation, so rates from
    # all transitions are summed at each R before taking the max.
    n_α, n_β, n_γ = 12, 6, 12
    α_vals = range(0, 2π, n_α + 1)[1:end-1]
    β_vals = range(0, π,  n_β + 1)[1:end-1]
    γ_vals = range(0, 2π, n_γ + 1)[1:end-1]
    R_arr = vec([from_euler_angles(α, β, γ)
                 for α in α_vals, β in β_vals, γ in γ_vals])

    # Sweep over DM masses. For each mass, sum rates over all transitions at
    # each orientation, then record the max/min/mean over orientations.
    # Each mass point is independent so the outer loop is threaded.
    mχ_vals_MeV = 10 .^ range(log10(1.0), log10(1000.0), 100)
    results = Vector{Tuple{Float64, Float64, Float64, Float64}}(undef, length(mχ_vals_MeV))

    Threads.@threads for i in eachindex(mχ_vals_MeV)
        mχ_MeV = mχ_vals_MeV[i]
        mχ = mχ_MeV * VSDM.MeV
        Γ_total = zeros(length(R_arr))
        for (pfq, dE) in zip(pfq_list, transition_energies)
            model = VSDM.ModelDMSM(0, mχ, mSM, dE * VSDM.eV)
            Γ_total .+= with_logger(NullLogger()) do
                VSDM.rate(R_arr, model, pfv, pfq)
            end
        end
        results[i] = (mχ_MeV, maximum(Γ_total), minimum(Γ_total),
                      sum(Γ_total) / length(Γ_total))
        println("mχ = (round(mχ_MeV, digits=2)) MeV  Γ_max = (results[i][2])")
    end

    # Save to CSV.
    output_path = joinpath(spherical_dir, "max_rates.csv")
    open(output_path, "w") do io
        println(io, "# vmax_c=(v_max),qmax_eV=(pfq_list[1].radial_basis.umax)")
        println(io, "mchi_MeV,rate_max,rate_min,rate_mean")
        for (mχ_MeV, r_max, r_min, r_mean) in results
            println(io, "mχ_MeV,r_max,r_min,r_mean")
        end
    end
    println("Saved to output_path")


end
"""
