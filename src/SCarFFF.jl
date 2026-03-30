# This is a small package wrapper so that we can precompile the form factor code.
# Also looks really cool to see SCarFFF precompiling.

module SCarFFF

using PrecompileTools
using HDF5
using CUDA

# Import the utility modules.
include("utils/FastPowers.jl")
include("utils/SparseTensors.jl")

# Next import the three methods. SCarFFF = Spherical, Cartesian, and Fourier Form Factors.
include("form_factors/spherical/SphericalFormFactor.jl")
include("form_factors/fft/FFTFormFactor.jl")
include("form_factors/cartesian/CartesianFormFactor.jl")

# Import the functions to compute the form factors from each module.
using .SphericalFormFactor: compute_spherical_form_factor
using .FFTFormFactor: compute_fft_form_factor
using .CartesianFormFactor: compute_cartesian_form_factor

# And export them for other modules to use.
export compute_spherical_form_factor, compute_fft_form_factor, compute_cartesian_form_factor

function write_mock_molecular_data_file(file_path::String, ::Type{T}) where {T<:AbstractFloat}
    """
    Write a minimal HDF5 molecular data file for precompilation.
    Creates a H2 molecule with bond length 1.4 Bohr (~0.74 Angstrom),
    each with a 1s orbital, unit transition matrix, and 1 eV transition energy.

    # Arguments:
    - file_path::String: The path where the HDF5 file will be written.
    - T::Type: The floating point type to use.
    """
    h5open(file_path, "w") do io
        # Write the basis set.
        HDF5.attributes(io)["basis"] = "6-31g*"

        # Write the atom data. This is just two H atoms separated along the x-axis by 1.4 Bohr.
        # We transpose the data to match Python's row-major format since restore_python_hdf5_order will transpose it back.
        io["atom_symbols"] = ["H", "H"]
        io["atom_coordinates"] = permutedims(T[-0.7 0.0 0.0; 0.7 0.0 0.0], (2, 1))

        # Write  the orbital labels. This is just two 1s orbitals.
        io["ao_labels"] = ["0 H 1s", "1 H 1s"]

        # Write the transition data. Just single transition at 1.0 eV with identity transition matrix.
        # We have to transpose again here because of python-Julia weirdness.
        io["energies_ev"] = T[1.0]
        io["d_ij_state_1"] = permutedims(T[1.0 0.0; 0.0 1.0], (2, 1))
    end
end

# Begin the precompilation block.
@setup_workload begin
    # Precompile the fast power functions.
    @compile_workload begin
        FastPowers.fast_i_pow(3, Float64)
        FastPowers.fast_neg1_pow(3, Float64)
        FastPowers.pow_int(1.234, 4)
    end

    # Precompile the spherical form factor CPU path.
    @compile_workload begin
        T = Float32

        # Create a temporary directory and write a minimal molecular data file.
        tmp_dir = mktempdir()
        mol_path = joinpath(tmp_dir, "mock_molecule.h5")
        write_mock_molecular_data_file(mol_path, T)

        # Read the molecular data from the HDF5 file.
        mol = SphericalFormFactor.ReadBasisSet.get_molecular_data(mol_path, precision=T)

        # Silence verbose output from the computation functions.
        redirect_stdout(devnull) do
            # Construct the pair coefficients and b coefficients.
            M_ij, sigma_ij, R_ij, R_ij_mod, R_ij_hat = SphericalFormFactor.ConstructPairCoefficients.construct_pair_coefficients(mol)
            b_A, b_B, b_C = SphericalFormFactor.ConstructbCoefficients.construct_b_coefficients(mol, R_ij)

            # Construct the D tensor from the molecular data.
            D_ij = SphericalFormFactor.ConstructDTensor.construct_D_tensor(mol, M_ij, sigma_ij, b_A, b_B, b_C)

            # Precompute Gaunt and A tensors, and save them to temporary files.
            gaunt_path = joinpath(tmp_dir, "gaunt_precompile.h5")
            A_tensor_path = joinpath(tmp_dir, "A_tensor_precompile.h5")
            lambda_max = 12
            l_max = 6
            SphericalFormFactor.PrecomputeGaunt.precompute_gaunt_coefficients(lambda_max, l_max + lambda_max, l_max, gaunt_path, T)
            SphericalFormFactor.PrecomputeATensor.precompute_A_tensor(lambda_max, A_tensor_path, T)

            # Contract D with A to get the W tensor.
            W_ij = SphericalFormFactor.ConstructWTensor.construct_W_tensor(D_ij, A_tensor_path, threshold = T(0))

            # Define a minimal grid for precompilation.
            q_grid = T[0.0, 0.1]
            theta_grid = T[0.0]
            phi_grid = T[0.0]
            transition_matrices = [mol.transition_matrices[1]]

            # Construct the R tensor.
            R_tensor = SphericalFormFactor.ConstructRTensor.construct_R_tensor(
                W_ij,
                sigma_ij,
                R_ij_mod,
                R_ij_hat,
                q_grid,
                l_max,
                gaunt_path,
                transition_matrices,
                mol.cartesian_term_to_orbital;
                threshold = T(0),
            )

            # Contract with the spherical harmonics to get the final form factor.
            _ = SphericalFormFactor.ContractSphericalGrid.contract_spherical_grid(R_tensor, theta_grid, phi_grid)
        end
    end

    # Precompile the GPU path if a GPU is available and kernel compilation works.
    # We test actual kernel compilation since CUDA.functional() can return true even when
    # kernel compilation fails due to incompatible drivers or missing hardware.
    gpu_available = false
    try
        if SphericalFormFactor.CUDA.functional()
            # Test if we can actually compile a trivial GPU kernel.
            SphericalFormFactor.CUDA.@sync SphericalFormFactor.CUDA.ones(Float32, 1)
            gpu_available = true
        end
    catch
        # GPU not available or kernel compilation failed.
        gpu_available = false
    end

    if gpu_available
        try
            @compile_workload begin
                T = Float32

                # Create a temporary directory and write a minimal molecular data file.
                tmp_dir = mktempdir()
                mol_path = joinpath(tmp_dir, "mock_molecule.h5")
                write_mock_molecular_data_file(mol_path, T)

                # Read the molecular data from the HDF5 file.
                mol = SphericalFormFactor.ReadBasisSet.get_molecular_data(mol_path, precision=T)

                # Silence verbose output from the computation functions.
                redirect_stdout(devnull) do
                    # Construct pair coefficients and b coefficients.
                    M_ij, sigma_ij, R_ij, R_ij_mod, R_ij_hat = SphericalFormFactor.ConstructPairCoefficients.construct_pair_coefficients(mol)
                    b_A, b_B, b_C = SphericalFormFactor.ConstructbCoefficients.construct_b_coefficients(mol, R_ij)

                    # Construct the D tensor from the molecular data.
                    D_ij = SphericalFormFactor.ConstructDTensor.construct_D_tensor(mol, M_ij, sigma_ij, b_A, b_B, b_C)

                    # Precompute Gaunt and A tensors, and save them to temporary files.
                    gaunt_path = joinpath(tmp_dir, "gaunt_precompile.h5")
                    A_tensor_path = joinpath(tmp_dir, "A_tensor_precompile.h5")
                    lambda_max = 4
                    l_max = 6
                    SphericalFormFactor.PrecomputeGaunt.precompute_gaunt_coefficients(lambda_max, l_max + lambda_max, l_max, gaunt_path, T)
                    SphericalFormFactor.PrecomputeATensor.precompute_A_tensor(lambda_max, A_tensor_path, T)

                    # Contract D with A to get the W tensor.
                    W_ij = SphericalFormFactor.ConstructWTensor.construct_W_tensor(D_ij, A_tensor_path, threshold = T(0))

                    # Define a minimal grid for precompilation.
                    q_grid = T[0.0, 0.1]
                    l_max = 6
                    transition_matrices = [mol.transition_matrices[1]]

                    # Construct the R tensor components on the GPU.
                    lambda_max = maximum(W_ij.lambda)
                    n_max = W_ij.n_max
                    R_lm_pos, R_lm_neg = SphericalFormFactor.ConstructRTensorGPU.construct_R_tensor_gpu(
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
                        threshold = T(0),
                    )

                    # Contract with the spherical harmonics on the GPU to get the final form factor.
                    theta_grid = T[0.0]
                    phi_grid = T[0.0]
                    _ = SphericalFormFactor.ContractSphericalGridGPU.contract_spherical_grid_gpu(R_lm_pos, R_lm_neg, theta_grid, phi_grid)
                end
            end
        catch
            # Ignore any GPU/toolchain issues during precompile (e.g., no GPU, old CUDA version, etc.).
        end
    end

    # Precompile the Cartesian form factor CPU path.
    @compile_workload begin
        T = Float32

        # Create a temporary directory and write a minimal molecular data file.
        tmp_dir = mktempdir()
        mol_path = joinpath(tmp_dir, "mock_molecule.h5")
        write_mock_molecular_data_file(mol_path, T)

        # Read the molecular data from the HDF5 file.
        mol = CartesianFormFactor.ReadBasisSet.get_molecular_data(mol_path, precision=T)

        # Silence verbose output from the computation functions.
        redirect_stdout(devnull) do
            # Construct pair coefficients and b coefficients.
            M_ij, sigma_ij, R_ij, R_ij_mod, R_ij_hat = CartesianFormFactor.ConstructPairCoefficients.construct_pair_coefficients(mol)
            b_A, b_B, b_C = CartesianFormFactor.ConstructbCoefficients.construct_b_coefficients(mol, R_ij)

            # Define a minimal Cartesian q-grid for precompilation.
            q_vals = T[0.0, 0.1]

            # Construct the V tensors.
            V_x, V_y, V_z, nonzero_pairs = CartesianFormFactor.ConstructVTensors.construct_V_tensors(
                mol,
                M_ij,
                sigma_ij,
                R_ij,
                b_A,
                b_B,
                b_C,
                q_vals,
                q_vals,
                q_vals;
                threshold = T(0)
            )

            # Contract the V tensors to get the form factor on the Cartesian grid.
            transition_matrices = [mol.transition_matrices[1]]
            _ = CartesianFormFactor.ContractCartesianGrid.contract_cartesian_grid(
                V_x,
                V_y,
                V_z,
                nonzero_pairs,
                M_ij,
                transition_matrices,
                mol.cartesian_term_to_orbital;
                threshold = T(0)
            )
        end
    end

    # Precompile the Cartesian form factor GPU path.
    if gpu_available
        try
            @compile_workload begin
                T = Float32

                # Create a temporary directory and write a minimal molecular data file.
                tmp_dir = mktempdir()
                mol_path = joinpath(tmp_dir, "mock_molecule.h5")
                write_mock_molecular_data_file(mol_path, T)

                # Read the molecular data from the HDF5 file.
                mol = CartesianFormFactor.ReadBasisSet.get_molecular_data(mol_path, precision=T)

                # Silence verbose output from the computation functions.
                redirect_stdout(devnull) do
                    # Construct pair coefficients and b coefficients.
                    M_ij, sigma_ij, R_ij, R_ij_mod, R_ij_hat = CartesianFormFactor.ConstructPairCoefficients.construct_pair_coefficients(mol)
                    b_A, b_B, b_C = CartesianFormFactor.ConstructbCoefficients.construct_b_coefficients(mol, R_ij)

                    # Define a minimal Cartesian q-grid for precompilation.
                    q_vals = T[0.0, 0.1]

                    # Construct the V tensors.
                    V_x, V_y, V_z, nonzero_pairs = CartesianFormFactor.ConstructVTensors.construct_V_tensors(
                        mol,
                        M_ij,
                        sigma_ij,
                        R_ij,
                        b_A,
                        b_B,
                        b_C,
                        q_vals,
                        q_vals,
                        q_vals;
                        threshold = T(0)
                    )

                    # Contract the V tensors to get the form factor on the Cartesian grid using GPU.
                    transition_matrices = [mol.transition_matrices[1]]
                    _ = CartesianFormFactor.ContractCartesianGridGPU.contract_cartesian_grid_gpu(
                        V_x,
                        V_y,
                        V_z,
                        nonzero_pairs,
                        M_ij,
                        transition_matrices,
                        mol.cartesian_term_to_orbital;
                        threshold = T(0)
                    )
                end
            end
        catch
            # Ignore any GPU/toolchain issues during precompile (e.g., no GPU, old CUDA version, etc.).
        end
    end

    # Precompile the FFT form factor path.
    @compile_workload begin
        T = Float32

        # Create a temporary directory and write a minimal molecular data file.
        tmp_dir = mktempdir()
        mol_path = joinpath(tmp_dir, "mock_molecule.h5")
        write_mock_molecular_data_file(mol_path, T)

        # Read the molecular data from the HDF5 file.
        mol = FFTFormFactor.ReadBasisSet.get_molecular_data(mol_path, precision=T)

        # Silence verbose output from the computation functions.
        redirect_stdout(devnull) do
            # Define a minimal q-grid for precompilation.
            q_vals = T[0.0, 0.1]

            # Construct the transition density in real space (CPU).
            transition_matrices = [mol.transition_matrices[1]]
            transition_density, r_lim = FFTFormFactor.ConstructTransitionDensity.construct_transition_density(
                mol,
                transition_matrices,
                q_vals,
                q_vals,
                q_vals
            )

            # Perform the FFT to get the form factor in momentum space.
            N_grid = [length(q_vals), length(q_vals), length(q_vals)]
            _ = FFTFormFactor.PerformFFT.perform_fft(transition_density, r_lim, N_grid)

            # If a GPU is available, trigger the GPU transition density kernel to warm up the CUDA path.
            if gpu_available
                try
                    transition_density_gpu, r_lim_gpu = FFTFormFactor.ConstructTransitionDensityGPU.construct_transition_density_gpu(
                        mol,
                        transition_matrices,
                        q_vals,
                        q_vals,
                        q_vals
                    )
                    # Ensure that the GPU array is actually computed.
                    sum(transition_density_gpu)
                    N_grid = [length(q_vals), length(q_vals), length(q_vals)]
                    _ = FFTFormFactor.PerformFFTGPU.perform_fft_gpu(transition_density_gpu, r_lim_gpu, N_grid)
                catch
                    # Ignore GPU warmup failures during precompile.
                end
            end
        end
    end
end

end
