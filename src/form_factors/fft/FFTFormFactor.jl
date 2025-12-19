module FFTFormFactor

using CUDA

include("../../utils/ReadBasisSet.jl")
include("ConstructIntegrand.jl")
include("ConstructIntegrandGPU.jl")
include("PerformFFT.jl")
include("PerformFFTGPU.jl")

using .ReadBasisSet: get_molecular_data, MoleculeData
using .ConstructTransitionDensity: construct_transition_density
using .ConstructTransitionDensityGPU: construct_transition_density_gpu
using .PerformFFT: perform_fft, check_parseval_theorem
using .PerformFFTGPU: perform_fft_gpu

export compute_fft_form_factor

function compute_fft_form_factor(
        qx_grid::Vector{T},
        qy_grid::Vector{T},
        qz_grid::Vector{T},
        td_dft_path::String;
        transition_indices::Vector{Int} = [1],
        check_parseval::Bool = false,
        use_gpu::Bool = false
    )::Tuple{Array{Complex{T}, 4}, Array{T, 4}, Vector{T}, Vector{T}} where {T<:AbstractFloat}
    """
    Compute the form factor using the FFT method.

    This method computes the form factor by first constructing the transition density:

        ρ(r) = ϕ_f(r) T_{fi} ϕ_i(r),

    on a Cartesian grid in real space, followed by a 3D FFT to obtain the form factor in momentum space.

    # Arguments:
    - qx_grid::Vector{T}: The qx grid in momentum space. Units of keV.
    - qy_grid::Vector{T}: The qy grid in momentum space. Units of keV.
    - qz_grid::Vector{T}: The qz grid in momentum space. Units of keV.
    - td_dft_path::String: The path to the TD-DFT results HDF5 file.
    - transition_indices::Vector{Int}: The transition indices to compute (default: [1]).
    - check_parseval::Bool: Whether to check that Parseval's theorem holds (default: false).
    - use_gpu::Bool: Whether to use GPU acceleration (default: false).

    # Returns:
    - form_factors::Array{Complex{T}, 4}: The form factors in momentum space with shape (n_transitions, n_qx, n_qy, n_qz).
    - transition_densities::Array{T, 4}: The transition densities in real space with shape (n_transitions, n_x, n_y, n_z).
    - r_lim::Vector{T}: The limits of the grid in real space, in Angstroms.
    - transition_energies_eV::Vector{T}: The transition energies in eV.
    """

    # Construct the MoleculeData structure.
    mol = get_molecular_data(td_dft_path; precision = T)

    # Get the transition matrices for the specified transitions.
    transition_matrices = [mol.transition_matrices[idx] for idx in transition_indices]

    # Construct the transition densities on a Cartesian grid.
    N_grid = [length(qx_grid), length(qy_grid), length(qz_grid)]

    if use_gpu
        transition_densities_gpu, r_lim = construct_transition_density_gpu(
            mol,
            transition_matrices,
            qx_grid,
            qy_grid,
            qz_grid
        )
        CUDA.synchronize()

        # Perform the FFT on the GPU.
        form_factors_gpu = perform_fft_gpu(transition_densities_gpu, r_lim, N_grid)
        CUDA.synchronize()

        # Transfer the results to CPU.
        transition_densities = Array(transition_densities_gpu)
        form_factors = Array(form_factors_gpu)
        CUDA.synchronize()

        # Check Parseval's theorem if requested.
        if check_parseval
            for t_idx in 1:length(transition_matrices)
                check_parseval_theorem(
                    view(transition_densities, t_idx, :, :, :),
                    view(form_factors, t_idx, :, :, :),
                    r_lim, N_grid
                )
            end
        end
    else
        # CPU path.
        transition_densities, r_lim = construct_transition_density(
            mol,
            transition_matrices,
            qx_grid,
            qy_grid,
            qz_grid
        )

        # Store a copy of the transition densities for the Parseval's theorem check if needed.
        transition_densities_copy = check_parseval ? copy(transition_densities) : nothing

        # Perform the FFT on CPU.
        form_factors = perform_fft(transition_densities, r_lim, N_grid)

        # Check Parseval's theorem if requested.
        if check_parseval
            for t_idx in 1:length(transition_matrices)
                check_parseval_theorem(
                    view(transition_densities_copy, t_idx, :, :, :),
                    view(form_factors, t_idx, :, :, :),
                    r_lim, N_grid
                )
            end
        end
    end

    # Extract the transition energies for the requested transitions.
    transition_energies_eV = [T(mol.transition_energies_eV[idx]) for idx in transition_indices]

    return form_factors, transition_densities, r_lim, transition_energies_eV
end

end
