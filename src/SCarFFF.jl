# This is a small package wrapper so that we can precompile the form factor code.
# Also looks really cool to see SCarFFF precompiling.

module SCarFFF

using PrecompileTools
using HDF5
using CUDA

# Import the utility modules.
include("utils/FastPowers.jl")
include("utils/ThreadChunks.jl")
include("utils/SparseTensors.jl")
include("utils/StageTimings.jl")

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

        # Write the transition data. Just a single transition at 1.0 eV, with one occupied and one
        # virtual orbital, whose amplitudes and coefficients give an identity transition matrix.
        # We have to transpose again here because of python-Julia weirdness.
        io["Delta_E_eV"] = T[1.0]
        io["C_o"] = permutedims(Complex{T}[1.0; 0.0;;], (2, 1))
        io["C_v"] = permutedims(Complex{T}[0.0; 1.0;;], (2, 1))
        io["X_state_1"] = permutedims(Complex{T}[1.0;;], (2, 1))
        io["Y_state_1"] = permutedims(Complex{T}[0.0;;], (2, 1))
    end
end

function run_mock_spherical_cpu(::Type{T}) where {T<:AbstractFloat}
    """
    Run the molecular spherical form factor on the mock molecule, for precompilation.

    # Arguments:
    - T::Type{<:AbstractFloat}: The floating-point type to run at.
    """

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
        lambda_max = 8
        l_max = 6
        SphericalFormFactor.PrecomputeGaunt.precompute_gaunt_coefficients(lambda_max, l_max + lambda_max, l_max, gaunt_path, T)
        SphericalFormFactor.PrecomputeATensor.precompute_A_tensor(lambda_max, A_tensor_path, T)

        # Contract D with A to get the W tensor.
        W_ij = SphericalFormFactor.ConstructWTensor.construct_W_tensor(D_ij, A_tensor_path, threshold = T(0))

        # Define a minimal grid for precompilation.
        q_grid = T[0.0, 0.1]
        theta_grid = T[0.0]
        phi_grid = T[0.0]
        T_gs = [mol.T_gs[1]]

        # Construct the R tensor.
        R_tensor = SphericalFormFactor.ConstructRTensor.construct_R_tensor(
            W_ij,
            sigma_ij,
            R_ij_mod,
            R_ij_hat,
            q_grid,
            l_max,
            gaunt_path,
            T_gs,
            mol.cartesian_term_to_orbital,
        )

        # Contract with the spherical harmonics to get the final form factor.
        _ = SphericalFormFactor.ContractSphericalGrid.contract_spherical_grid(R_tensor, theta_grid, phi_grid)

        # The additional densities needed by Ewald.
        if T === Float64
            density_l_max = 4
            n_theta, n_phi = max.(3, SphericalFormFactor.default_angular_grid(density_l_max))
            density_theta = collect(range(zero(T), T(π); length = n_theta))
            density_phi = collect(range(zero(T), T(2π); length = n_phi))
            _ = SphericalFormFactor.compute_spherical_form_factor_with_densities(
                T[0.0, 5.0, 10.0], density_theta, density_phi, density_l_max, mol_path;
                transition_indices = [1], force_recomputation = false, threshold = T(1.0e-2),
                use_gpu = false, need_grid = false, need_R = true, need_f_sq_lm = true,
                need_short_range = true)

            _ = SphericalFormFactor.compute_spherical_form_factor_with_densities(
                T[0.0, 0.1], density_theta, density_phi, density_l_max, mol_path;
                transition_indices = [1], force_recomputation = false, threshold = T(1.0e-2),
                use_gpu = false, need_grid = false, need_R = true, need_f_sq_lm = false)
        end
    end

    return nothing
end

function run_mock_crystal(::Type{T}) where {T<:AbstractFloat}
    """
    Run the crystal half of the pipeline on a minimal two-molecule cell, for precompilation.

    # Arguments:
    - T::Type{<:AbstractFloat}: The floating-point type to run at.
    """

    SFF = SphericalFormFactor
    BH = SFF.BlochHamiltonian
    EW = SFF.Ewald
    SVector = BH.SVector
    rotor = BH.Quaternionic.rotor

    l_max = 2
    n_keys = (l_max + 1)^2
    q_grid = collect(range(zero(T), T(6); length = 33))
    n_q = length(q_grid)
    q_bc = T(1.0e-4)

    # One neutral, spherically symmetric density, carried entirely by the (ℓ, m) = (0, 0) mode.
    shape(q) = T(0.8) * q * q * exp(-q * q / 2)
    density = zeros(Complex{T}, 1, n_q, n_keys)
    for (q_idx, q) in enumerate(q_grid)
        density[1, q_idx, 1] = shape(q)
    end
    density_bc = zeros(Complex{T}, 1, 1, n_keys)
    density_bc[1, 1, 1] = shape(q_bc)

    # Two transitions per molecule, so Δ carries all three (s, t) pairs.
    conformer_R = vcat(density, T(0.7) .* density)
    conformer_Delta = vcat(T(0.6) .* density, T(-0.35) .* density, T(0.2) .* density)
    conformer_Xi = T(-0.9) .* density
    conformer_R_bc = vcat(density_bc, T(0.7) .* density_bc)
    conformer_Delta_bc = vcat(T(0.6) .* density_bc, T(-0.35) .* density_bc,
                              T(0.2) .* density_bc)
    conformer_Xi_bc = T(-0.9) .* density_bc

    # Two molecules in the cell, the second rotated, so the unlike-molecule branches run too.
    lattice = SFF.build_crystal_lattice(T[4 0 0; 0 4.5 0; 0 0 5], T)
    images = [BH.CrystalImage{T}(1, rotor(T[1, 0, 0, 0]), one(T), SVector{3, T}(0, 0, 0)),
              BH.CrystalImage{T}(1, rotor(T[0, 1, 0, 0]), one(T), SVector{3, T}(2, 2.25, 0))]
    basis = SFF.build_excitation_basis(images, [T[2.0, 2.5]])

    rotated_R, rotated_Delta, rotated_Xi = SFF.rotate_crystal_R_tensors(
        [conformer_R], [conformer_Delta], [conformer_Xi], images, basis, l_max)
    rotated_R_bc, rotated_Delta_bc, rotated_Xi_bc = SFF.rotate_crystal_R_tensors(
        [conformer_R_bc], [conformer_Delta_bc], [conformer_Xi_bc], images, basis, l_max)

    n_theta, n_phi = max.(3, SFF.default_angular_grid(l_max))
    theta_grid = collect(range(zero(T), T(π); length = n_theta))
    phi_grid = collect(range(zero(T), T(2π); length = n_phi))
    operations = SFF.derive_symmetry_operations(images, lattice)
    stars = SFF.build_stars(theta_grid, phi_grid, operations)

    mktempdir() do tmp_dir
        gaunt_path = joinpath(tmp_dir, "coupling_gaunt.h5")
        SFF.precompute_gaunt_coefficients(l_max, 2 * l_max, l_max, gaunt_path, T)

        # Both splits are chosen the way the crystal job chooses them.
        n_stars = length(stars.irreducible_directions)
        # J's cost carries the form factor term, so it needs a Cartesian pair count.
        SFF.choose_ewald_parameters(lattice, length(images), T(1.0e-3), SFF.J_TRUNCATION;
            q_reach = q_grid[end], cost = SFF.J_COST, n_transitions = basis.n_transitions,
            l_max = l_max, n_q = n_q, n_stars = n_stars, n_pair = 15000)
        SFF.choose_ewald_parameters(lattice, length(images), T(1.0e-3), SFF.D_TRUNCATION;
            q_reach = q_grid[end], cost = SFF.D_COST, n_transitions = basis.n_transitions,
            l_max = l_max, n_q = n_q, n_stars = n_stars)

        # Fixed by hand from here, so retuning the cost model cannot blow up the cell count.
        J_parameters = EW.EwaldParameters{T}(T(0.45), T(5), T(4), T(1.0e-8))
        D_parameters = EW.EwaldParameters{T}(T(0.5), T(4.5), T(4.3), T(1.0e-8))
        cells = SFF.enumerate_neighbour_cells(
            lattice, max(J_parameters.R_max, D_parameters.R_max) +
                     SFF.image_translation_span(basis))

        long_range = SFF.build_ewald_long_range(
            rotated_R, rotated_R_bc,
            [basis.images[basis.image_of[i]].tau for i in 1:length(basis.Delta_E_monomer)],
            basis.image_of, lattice, q_grid, q_bc, l_max, J_parameters)

        couplings, D_short = SFF.compute_crystal_corrections(
            rotated_R, rotated_Delta, rotated_Xi, basis, cells, q_grid, l_max, gaunt_path;
            J_parameters = J_parameters, D_parameters = D_parameters)
        SFF.subtract_self_term!(couplings, long_range.J_self)

        D_long = SFF.compute_ewald_D(
            rotated_Delta, rotated_Xi, rotated_Delta_bc, rotated_Xi_bc,
            q_grid, q_bc, basis.n_transitions, l_max, long_range, D_parameters)
        SFF.set_D_couplings!(basis, D_short .+ D_long)

        # The reference scales that turn the requested relative radial error into eV.
        SFF.short_range_scales(couplings, D_short .+ D_long, basis, lattice, long_range)

        # The direct lattice sum, which runs when a job asks for no Ewald split.
        SFF.compute_crystal_corrections(
            rotated_R, rotated_Delta, rotated_Xi, basis, cells, q_grid, l_max, gaunt_path)

        # The band range, the energy axis it feeds, and the coherent loop over stars.
        energy_low, energy_high, energy_margin = SFF.band_energy_range(
            basis, lattice, q_grid, stars, couplings, long_range;
            star_stride = 1, q_stride = 1)
        energy_grid = SFF.build_energy_grid(energy_low - energy_margin,
                                            energy_high + energy_margin, T(0.05))
        # With and without the energy axis, and with and without the saved grid. Each is a
        # separate branch, so each needs compiling.
        for energy in (energy_grid, nothing), grid in (false, true)
            SFF.compute_coherent_unit_cell_f_sq_lm(
                rotated_R, basis, lattice, q_grid, theta_grid, phi_grid, l_max, stars;
                need_grid = grid, couplings = couplings, long_range = long_range,
                energy_grid = energy)
        end
        for grid in (false, true)
            SFF.compute_incoherent_unit_cell_f_sq_lm(
                rotated_R, basis, q_grid, theta_grid, phi_grid, l_max; need_grid = grid)
        end
    end

    return nothing
end

# Begin the precompilation block.
@setup_workload begin
    # Precompile the fast power functions.
    @compile_workload begin
        FastPowers.fast_i_pow(3, Float64)
        FastPowers.fast_neg1_pow(3, Float64)
        FastPowers.pow_int(1.234, 4)
    end

    # Precompile the crystal path.
    @compile_workload begin
        redirect_stdout(devnull) do
            run_mock_crystal(Float64)
        end
    end

    # Precompile the spherical form factor CPU path.
    @compile_workload begin
        run_mock_spherical_cpu(Float32)
        run_mock_spherical_cpu(Float64)
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
                    T_gs = [mol.T_gs[1]]

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
                        T_gs,
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
            T_gs = [mol.T_gs[1]]
            _ = CartesianFormFactor.ContractCartesianGrid.contract_cartesian_grid(
                V_x,
                V_y,
                V_z,
                nonzero_pairs,
                M_ij,
                T_gs,
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
                    T_gs = [mol.T_gs[1]]
                    _ = CartesianFormFactor.ContractCartesianGridGPU.contract_cartesian_grid_gpu(
                        V_x,
                        V_y,
                        V_z,
                        nonzero_pairs,
                        M_ij,
                        T_gs,
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
            T_gs = [mol.T_gs[1]]
            transition_density, r_lim = FFTFormFactor.ConstructTransitionDensity.construct_transition_density(
                mol,
                T_gs,
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
                        T_gs,
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
