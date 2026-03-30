# This module contains functions to compute the form factor on a Cartesian grid.

module CartesianFormFactor

include("../../utils/ReadBasisSet.jl")
include("../common/ConstructPairCoefficients.jl")
include("../common/ConstructbCoefficients.jl")
include("ProbabilistsHermite.jl")
include("ConstructVTensors.jl")
include("ContractCartesianGrid.jl")
include("ContractCartesianGridGPU.jl")

using CUDA
using .ReadBasisSet: get_molecular_data, MoleculeData
using .ConstructPairCoefficients: construct_pair_coefficients
using .ConstructbCoefficients: construct_b_coefficients
using .ConstructVTensors: construct_V_tensors
using .ContractCartesianGrid: contract_cartesian_grid
using .ContractCartesianGridGPU: contract_cartesian_grid_gpu

export compute_cartesian_form_factor

# Conversion factor from keV to inverse Angstroms.
const KEV_TO_INV_ANGSTROM = 1 / 1.973269803

function compute_cartesian_form_factor(
    q_x_vals::Vector{T},
    q_y_vals::Vector{T},
    q_z_vals::Vector{T},
    td_dft_path::String;
    transition_indices::Vector{Int} = [1],
    threshold::T = zero(T),
    use_gpu::Bool = false,
    need_grid::Bool = true,
    need_V::Bool = true
)::Tuple{Union{Nothing, Tuple{Array{Complex{T}, 2}, Array{Complex{T}, 2}, Array{Complex{T}, 2}}}, Union{Nothing, Array{Complex{T}, 4}}, Vector{T}} where {T<:AbstractFloat}
    """
    Compute the Cartesian form factor f_s(q_x, q_y, q_z) on a Cartesian grid using the
    separable tensor contraction method.

    # Arguments:
    - q_x_vals::Vector{T}: The q_x values at which to evaluate the form factor, in units of keV.
    - q_y_vals::Vector{T}: The q_y values at which to evaluate the form factor, in units of keV.
    - q_z_vals::Vector{T}: The q_z values at which to evaluate the form factor, in units of keV.
    - td_dft_path::String: The path to the TD-DFT results JSON file.
    - transition_indices::Vector{Int}: The transition indices to compute (default: [1]).
    - threshold::T: A threshold value below which pairs satisfying (|M/M_max| < threshold) will be discarded (default: 0.0, so no discarding).
    - use_gpu::Bool: Whether to use the GPU for the final contraction step (default: false).
    - need_grid::Bool: Whether to contract the V tensors and compute the form factor on the grid (default: true).
    - need_V::Bool: Whether to save and return the V tensors (default: true).

    # Returns:
    - V_tensors::Union{Nothing, Tuple{Array{Complex{T}, 2}, Array{Complex{T}, 2}, Array{Complex{T}, 2}}}: The V_x, V_y, V_z tensors if need_V is true, otherwise nothing.
    - f_s::Union{Nothing, Array{Complex{T}, 4}}: The form factor on the (q_x, q_y, q_z) grid with shape (n_transitions, n_qx, n_qy, n_qz) if need_grid is true, otherwise nothing.
    - transition_energies_eV::Vector{T}: The transition energies in eV for the requested transitions.
    """

    # Construct MoleculeData structure.
    mol = get_molecular_data(td_dft_path; precision = T)

    # Convert q values from keV to inverse Angstroms.
    q_x_inv_ang = T(KEV_TO_INV_ANGSTROM) .* q_x_vals
    q_y_inv_ang = T(KEV_TO_INV_ANGSTROM) .* q_y_vals
    q_z_inv_ang = T(KEV_TO_INV_ANGSTROM) .* q_z_vals

    # Construct the pair coefficients.
    M_ij, sigma_ij, R_ij, _, _ = construct_pair_coefficients(mol)

    # Construct the b coefficients.
    b_A, b_B, b_C = construct_b_coefficients(mol, R_ij)

    # Extract the transition matrices for the requested transitions.
    transition_matrices = [mol.transition_matrices[idx] for idx in transition_indices]

    # Construct the V tensors.
    V_x, V_y, V_z, nonzero_pairs = construct_V_tensors(
        mol, M_ij, sigma_ij, R_ij, b_A, b_B, b_C,
        q_x_inv_ang, q_y_inv_ang, q_z_inv_ang;
        threshold = threshold
    )

    # Contract to get the form factors for the requested transitions (if needed).
    f_s = nothing
    if need_grid
        if use_gpu
            f_s = Array(contract_cartesian_grid_gpu(V_x, V_y, V_z, nonzero_pairs, M_ij, transition_matrices, mol.cartesian_term_to_orbital; threshold = threshold))
        else
            f_s = contract_cartesian_grid(V_x, V_y, V_z, nonzero_pairs, M_ij, transition_matrices, mol.cartesian_term_to_orbital; threshold = threshold)
        end
    end

    # Unset the V tensors if we don't want to save them.
    V_tensors = nothing
    if need_V
        V_tensors = (V_x, V_y, V_z)
    end

    # Extract the transition energies for the requested transitions.
    transition_energies_eV = [T(mol.transition_energies_eV[idx]) for idx in transition_indices]

    return V_tensors, f_s, transition_energies_eV
end

end