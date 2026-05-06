# This module contains functions to compute the DM scattering rate using VSDM.

module ComputeRates

using Base.Threads
using VectorSpaceDarkMatter
using Quaternionic
import LinearAlgebra: mul!
import Logging: NullLogger, with_logger

include("../../utils/DMModels.jl")
using .DMModels: spherical_halo, v_max

const VSDM = VectorSpaceDarkMatter
const G_CACHE = Ref{Union{Nothing, Matrix{Float64}}}(nothing)

function precompute_G_cache(R_arr, l_max::Int)
    """
    Precompute the real spherical harmonic rotation matrices used by VSDM. Each
    column is the flattened G matrix for one detector orientation.

    # Arguments:
    - R_arr: The detector rotation grid.
    - l_max::Int: The maximum angular momentum.

    # Returns:
    - G_cache::Matrix{Float64}: The cached G matrices with dimensions (N_G, N_rotations).
    """

    # Preallocate the cache arrays.
    N_rotations = length(R_arr)
    N_G = VSDM.WignerDsize(l_max)
    G_cache = Matrix{Float64}(undef, N_G, N_rotations)
    N_tasks = min(nthreads(), N_rotations) # Set the number of threads to use.
    null_logger = NullLogger()

    # Compute the G matrix for each rotation once, then reuse it for all masses.
    # Use the logger to suppress SphericalFunctions warnings.
    with_logger(null_logger) do
        @threads for task_idx in 1:N_tasks
            # Allocate a different vector for each thread since they are modified in place.
            D = VSDM.D_prep(l_max)
            G = zeros(Float64, N_G)

            @inbounds for rotation_idx in task_idx:N_tasks:N_rotations
                VSDM.D_matrices!(D, R_arr[rotation_idx])
                VSDM.G_matrices!(G, D)
                for G_idx in eachindex(G)
                    G_cache[G_idx, rotation_idx] = G[G_idx]
                end
            end
        end
    end

    return G_cache
end

function get_G_cache(R_arr, l_max::Int)
    """
    Get the cached G matrices for a rotation grid. The cache is shared across
    molecules in the same Julia process.

    # Arguments:
    - R_arr: The detector rotation grid.
    - l_max::Int: The maximum angular momentum.

    # Returns:
    - G_cache::Matrix{Float64}: The cached G matrices with dimensions (N_G, N_rotations).
    """

    # In a batch run, all molecules use the same l_max and rotation grid.
    if G_CACHE[] === nothing
        G_CACHE[] = precompute_G_cache(R_arr, l_max)
    end

    return G_CACHE[]
end

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

function compute_rates_by_orientation(
        flm::Array{T,3},
        q_grid::Vector{T},
        m_grid::Vector{T},
        transition_energies::Vector{T},
        N_rotations::Union{Nothing,Tuple{Int,Int,Int}}=nothing,
        model=spherical_halo(),
    )::Array{T, 2} where {T<:AbstractFloat}
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
    - rate_grid::Array{T, 2}: The dimensionless rate for each DM mass and detector orientation.
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

    # Get the cached G matrices, or compute them for the first time.
    G_cache = get_G_cache(R_arr, l_max)

    # Construct the projected form factor objects that VSDM expects.
    lm = VSDM.LM_vals(l_max)
    fs_list = Vector{VSDM.ProjectedF{Float64, typeof(q_basis)}}(undef, N_transitions)
    for transition_idx in 1:N_transitions
        fs_list[transition_idx] = VSDM.ProjectedF(Float64.(@view f_tophat[transition_idx, :, :]), lm, q_basis)
    end

    # Project the velocity distribution onto the tophat basis.
    # Here n_v is the number of tophat bins, so VSDM wants n_max = n_v - 1.
    v_tophat = VSDM.ProjectF(model,(n_v-1,l_max),v_basis)

    # Compute the dimensionless scattering rate here using VSDM.
    mSM = VSDM.mElec
    n_rotation_vals = length(R_arr)
    G_cache_transpose = transpose(G_cache)
    transition_energies_eV = Float64.(transition_energies)
    null_logger = NullLogger()
    rate_grid = zeros(T, length(m_grid), n_rotation_vals)

    # Sweep over DM masses. For each mass, sum rates over all transitions at
    # each orientation.
    @threads for i in eachindex(m_grid)
        mchi = Float64(m_grid[i]) * VSDM.MeV
        total_rate = zeros(Float64, n_rotation_vals)

        # Use the logger to suppress SpecialFunctions warnings.
        with_logger(null_logger) do
            for transition_idx in 1:N_transitions
                dm_model = VSDM.ModelDMSM(0, mchi, mSM, transition_energies_eV[transition_idx] * VSDM.eV)
                mcK = VSDM.get_mcalK(dm_model, v_tophat, fs_list[transition_idx]; use_measurements=false)
                mul!(total_rate, G_cache_transpose, mcK.K, 1.0, 1.0)
            end
        end

        rate_grid[i, :] .= T.(total_rate)
    end

    return rate_grid
end

function reduce_rate_grid(rate_grid::Array{T, 2}, m_grid::Vector{T}) where {T<:AbstractFloat}
    """
    Reduce a rate grid to max/min/mean statistics per DM mass.

    # Arguments:
    - rate_grid::Array{T, 2}: The rate grid with dimensions (n_masses, n_rotations).
    - m_grid::Vector{T}: The DM mass grid in MeV.

    # Returns:
    - results::Vector{NamedTuple{(:mchi_MeV, :rate_max, :rate_min, :rate_mean), NTuple{4,T}}}: The dimensionless rates for each DM mass.
    """

    N_rotations = size(rate_grid, 2)
    inv_N_rotations = inv(Float64(N_rotations))
    results = Vector{NamedTuple{(:mchi_MeV, :rate_max, :rate_min, :rate_mean), NTuple{4,T}}}(undef, length(m_grid))

    @threads for i in eachindex(m_grid)
        # Look at each mass slice.
        total_rate = @view rate_grid[i, :]

        rate_max = -Inf
        rate_min = Inf
        rate_sum = 0.0
        @inbounds for rate in total_rate
            # Compute the max/min/mean over orientations.
            rate_max = max(rate_max, rate)
            rate_min = min(rate_min, rate)
            rate_sum += rate
        end

        # Store the rates.
        results[i] = (
            mchi_MeV = m_grid[i],
            rate_max = T(rate_max),
            rate_min = T(rate_min),
            rate_mean = T(rate_sum * inv_N_rotations),
        )
    end

    return results
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

    rate_grid = compute_rates_by_orientation(
        flm,
        q_grid,
        m_grid,
        transition_energies,
        N_rotations,
        model,
    )
    # Reduce to mean/min/max for saving.
    return reduce_rate_grid(rate_grid, m_grid)
end

function combine_crystal_rate_grids(rate_grids::Vector{Array{T, 2}}, occupancies::Vector{T}, m_grid::Vector{T}) where {T<:AbstractFloat}
    """
    Combine conformer-set rate grids into a total crystal rate.

    # Arguments:
    - rate_grids::Vector{Array{T, 2}}: The rate grid for each conformer set.
    - occupancies::Vector{T}: The occupancy weight for each conformer set.
    - m_grid::Vector{T}: The DM mass grid in MeV.

    # Returns:
    - results::Vector{NamedTuple{(:mchi_MeV, :rate_max, :rate_min, :rate_mean), NTuple{4,T}}}: The occupancy-weighted rates for each DM mass.
    """

    # Initialise the total rate grid.
    total_rate_grid = zeros(T, size(rate_grids[1]))
    for (rate_grid, occupancy) in zip(rate_grids, occupancies)
        total_rate_grid .+= occupancy .* rate_grid
    end

    # Reduce to mean/min/max for saving.
    return reduce_rate_grid(total_rate_grid, m_grid)
end


end
