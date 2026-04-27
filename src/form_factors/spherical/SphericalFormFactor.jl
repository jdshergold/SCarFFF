# This module contains functions to compute the form factor on a spherical grid.

module SphericalFormFactor

include("../../utils/ReadBasisSet.jl")
include("../common/ConstructPairCoefficients.jl")
include("../common/ConstructbCoefficients.jl")
include("ConstructDTensor.jl")
include("../../precomputation/PrecomputeGaunt.jl")
include("../../precomputation/PrecomputeATensor.jl")
include("ConstructWTensor.jl")
include("ConstructRTensor.jl")
include("ContractSphericalGrid.jl")
include("ConstructRTensorGPU.jl")
include("ContractSphericalGridGPU.jl")
include("ConstructFLMTensor.jl")

using CUDA
using .ReadBasisSet: get_molecular_data, MoleculeData
using .ConstructPairCoefficients: construct_pair_coefficients
using .ConstructbCoefficients: construct_b_coefficients
using .ConstructDTensor: construct_D_tensor
using .PrecomputeGaunt: precompute_gaunt_coefficients
using .PrecomputeATensor: precompute_A_tensor
using .ConstructWTensor: construct_W_tensor
using .ConstructRTensor: construct_R_tensor
using .ContractSphericalGrid: contract_spherical_grid

using BenchmarkTools

export compute_spherical_form_factor

@inline function combine_gpu_R_tensor(R_pos, R_neg, l_max::Int)
    """
    Combine the m ≥ 0 and m < 0 GPU R tensors into a single prefactored R tensor keyed by:

        key(ℓ, m) = ℓ^2 + (ℓ + m) + 1,

    so that we can save this to disk.

    # Arguments:
    - R_pos::CuArray{Complex{T}}: The R tensor for m ≥ 0 with dimensions (n_transitions, n_q, n_keys_pos).
    - R_neg::CuArray{Complex{T}}: The R tensor for m < 0 with dimensions (n_transitions, n_q, n_keys_pos).
    - l_max::Int: The maximum angular mode requested.

    # Returns:
    - combined::Array{Complex{T}, 3}: The combined R tensor with dimensions (n_transitions, n_q, n_keys), keyed by key(ℓ, m) = ℓ^2 + (ℓ + m) + 1.
    """

    # Get the first two dimensions for the output array.
    n_transitions = size(R_pos, 1)
    n_q = size(R_pos, 2)

    # Move the R tensors to the CPU.
    R_pos_cpu = Array(R_pos)
    R_neg_cpu = Array(R_neg)

    # Allocate the combined R tensor.
    combined = Array{eltype(R_pos_cpu)}(undef, n_transitions, n_q, (l_max + 1)^2)

    @inbounds for l in 0:l_max
        # Compute the base keys for this ℓ.
        key_new_base = l * l + l + 1
        key_old_base = (l * (l + 1)) ÷ 2 + 1 # The single-sided key base.

        # Copy over the m = 0 component.
        combined[:, :, key_new_base] = R_pos_cpu[:, :, key_old_base]
        for m in 1:l
            # Compute the old and new keys.
            pos_key_old = key_old_base + m
            key_new = key_new_base + m

            # Copy over the m > 0 and m < 0 components.
            combined[:, :, key_new] = R_pos_cpu[:, :, pos_key_old]
            combined[:, :, key_new_base - m] = R_neg_cpu[:, :, pos_key_old]
        end
    end

    return combined
end

function compute_spherical_form_factor(
        q_grid::Vector{T},
        theta_grid::Vector{T},
        phi_grid::Vector{T},
        l_max::Int,
        td_dft_path::String;
        transition_indices::Vector{Int} = [1],
        force_recomputation::Bool = false,
        threshold::T = zero(T),
        use_gpu::Bool = false,
        need_grid::Bool = true,
        need_R::Bool = true,
        need_flm::Bool = true,
    )::Tuple{Union{Array{Complex{T}, 3}, Nothing}, Union{Array{Complex{T}, 4}, Nothing}, Union{Array{T, 3}, Nothing}, Vector{T}} where {T<:AbstractFloat}
    """
    Compute the spherical form factor f_s(q, θ, ϕ) on a spherical grid up to some angular mode ℓ_max.

    # Arguments:
    - q_grid::Vector{T}: The |q| values at which to evaulate the form factor, in units of keV.
    - theta_grid::Vector{T}: The θ values at which to evaluate the form factor, in radians.
    - phi_grid::Vector{T}: The ϕ values at which to evaluate the form factor, in radians.
    - l_max::Int: The maximum angular momentum quantum number ℓ to evaluate the form factor for.
    - td_dft_path::String: The path to the TD-DFT results JSON file.
    - transition_indices::Vector{Int}: The transition indices to compute (default: [1]).
    - force_recomputation::Bool: Whether to force the recomputation of Gaunt coefficients and the A tensor even if they exist (default: false).
    - threshold::T: A threshold value below which W tensor entries satisfying (|W/W_max| < threshold) will be discarded (default: 0.0, so no discarding).
    - use_gpu::Bool: Whether to use the GPU for steps beyond the W tensor (default: false).
    - need_grid::Bool: Whether to contract the R tensor to compute the form factor on the grid (default: true).
    - need_R::Bool: Whether to save the R tensor (default: true).
    - need_flm::Bool: Whether to compute the f_lm tensor (default: false).

    # Returns:
    - R_tensor::Union{Array{Complex{T}, 3}, Nothing}: The prefactored R tensor keyed by ℓ^2 + (ℓ + m) + 1, or nothing if not requested for saving.
    - f_s::Union{Array{Complex{T}, 4}, Nothing}: The form factor on the (q, θ, ϕ) grid with shape (n_transitions, n_q, n_θ, n_ϕ), or nothing if not requested.
    - f_lm::Union{Array{Complex{T}, 3}, Nothing}: The f_lm tensor, the ``squared'' verision of the R tensor, with shape (n_transitions, n_q, n_flm_keys), or nothing if not requested.
    - transition_energies_eV::Vector{T}: The transition energies in eV for the requested transitions.
    """

    # Construct MoleculeData structure.
    mol = get_molecular_data(td_dft_path; precision = T)

    # Determine lambda_max from the molecular data.
    lambda_max = 2 * maximum(mol.orbital_l)

    # Build the Gaunt coefficients and A tensor paths based on l_max and the specified precision.
    # Use @__DIR__ to make paths relative to this source file, not the working directory.
    type_suffix = T == Float32 ? "_f32" : "_f64"
    data_dir = joinpath(@__DIR__, "../../data")
    gaunt_path = joinpath(data_dir, "gaunt_coefficients", "gaunt_coefficients_lambda$(lambda_max)_L$(l_max + lambda_max)_l$(l_max)$(type_suffix).h5")
    A_tensor_path = joinpath(data_dir, "A_tensors", "A_tensor_n$(lambda_max)$(type_suffix).h5")

    # Precompute Gaunt coefficients if they do not already exist.
    if force_recomputation || !isfile(gaunt_path)
        precompute_gaunt_coefficients(lambda_max, l_max + lambda_max, l_max, gaunt_path, T)
    end

    # Precompute A tensor if it does not already exist.
    if force_recomputation || !isfile(A_tensor_path)
        precompute_A_tensor(lambda_max, A_tensor_path, T)
    end

    # Precompute Gaunt coefficients for the f_lm tensor if requested and if they do not already exist.
    if need_flm
        gaunt_flm_path = joinpath(data_dir, "gaunt_coefficients", "gaunt_coefficients_flm_lmax$(l_max)$(type_suffix).h5")
        if force_recomputation || !isfile(gaunt_flm_path)
            precompute_gaunt_coefficients(l_max, l_max, l_max, gaunt_flm_path, T)
        end
    end

    # Construct pair coefficients.
    M_ij, sigma_ij, R_ij, R_ij_mod, R_ij_hat = construct_pair_coefficients(mol)

    # Construct b coefficients.
    b_A, b_B, b_C = construct_b_coefficients(mol, R_ij)

    # Construct D tensor.
    D_ij = construct_D_tensor(mol, M_ij, sigma_ij, b_A, b_B, b_C)

    # Construct W tensor.
    threshold_T = T(threshold)
    W_ij = construct_W_tensor(D_ij, A_tensor_path, threshold = threshold_T)

    # Extract the transition matrices for the requested transitions.
    transition_matrices = [mol.transition_matrices[idx] for idx in transition_indices]

    R_lm = nothing
    f_s = nothing
    f_lm = nothing

    if use_gpu
        # Now construct the R tensor on the GPU.
        lambda_max = maximum(W_ij.lambda)
        n_max = W_ij.n_max

        R_lm_pos, R_lm_neg = ConstructRTensorGPU.construct_R_tensor_gpu(
            W_ij,
            sigma_ij,
            R_ij_mod,
            R_ij_hat,
            q_grid,
            l_max,
            lambda_max,
            n_max,
            gaunt_path,
            transition_matrices,
            mol.cartesian_term_to_orbital;
            threshold = threshold_T
        )

        if need_grid
            # Contract with the angular grid to get the form factor on the GPU.
            f_s_gpu = ContractSphericalGridGPU.contract_spherical_grid_gpu(R_lm_pos, R_lm_neg, theta_grid, phi_grid)
            f_s = Array(f_s_gpu)
            CUDA.unsafe_free!(f_s_gpu)
        end

        if need_R || need_flm
            R_lm = combine_gpu_R_tensor(R_lm_pos, R_lm_neg, l_max)
        end

        if need_flm
            # Construct the f_lm tensor on the CPU from the combined R tensor.
            f_lm = ConstructFLMTensor.construct_f_lm_tensor(R_lm, gaunt_flm_path)
        end

        # Unset R_lm if we only needed it as an intermediate for f_lm.
        if !need_R
            R_lm = nothing
        end

        CUDA.unsafe_free!(R_lm_pos)
        CUDA.unsafe_free!(R_lm_neg)
        CUDA.reclaim()

    else
        # Construct R tensor on the CPU.
        R_lm = construct_R_tensor(
            W_ij,
            sigma_ij,
            R_ij_mod,
            R_ij_hat,
            q_grid,
            l_max,
            gaunt_path,
            transition_matrices,
            mol.cartesian_term_to_orbital,
            threshold = threshold_T
        )

        if need_grid
            # Contract with the angular grid to get the form factor.
            f_s = contract_spherical_grid(R_lm, theta_grid, phi_grid)
        end

        if need_flm
            # Construct the f_lm tensor.
            f_lm = ConstructFLMTensor.construct_f_lm_tensor(R_lm, gaunt_flm_path)
        end

        # Unset R_lm if we don't want to save it.
        if !need_R
            R_lm = nothing
        end
        
    end

    # Extract the transition energies for the requested transitions.
    transition_energies_eV = [T(mol.transition_energies_eV[idx]) for idx in transition_indices]

    return R_lm, f_s, f_lm, transition_energies_eV
end

end
