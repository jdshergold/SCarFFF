using Test
using SCarFFF
using LinearAlgebra
using Quaternionic
using SphericalHarmonics
using StaticArrays

const SF = SCarFFF.SphericalFormFactor
const RT = SF.ConstructRTensor
const DF = SF.DensityFormFactors
const CJ = SF.CouplingJ
const BH = SF.BlochHamiltonian
const DM = SF.ReadBasisSet.DensityMatrices

function harmonic_values(direction::SVector{3, T}, l_max::Int) where {T<:AbstractFloat}
    """
    Evaluate complex spherical harmonics in the R-tensor key order.

    # Arguments:
    - direction::SVector{3,T}: Unit vector at which to evaluate the harmonics.
    - l_max::Int: The largest angular mode.

    # Returns:
    - Vector{Complex{T}}: Y_l^m(direction), keyed by l^2 + l + m + 1.
    """

    theta = acos(clamp(direction[3], -one(T), one(T)))
    phi = atan(direction[2], direction[1])
    cache = SphericalHarmonics.cache(l_max, SphericalHarmonics.FullRange)
    SphericalHarmonics.computePlmcostheta!(cache, theta, l_max)
    SphericalHarmonics.computeYlm!(cache, theta, phi, l_max)
    Y = SphericalHarmonics.getY(cache)
    return Complex{T}[Y[(l, m)] for l in 0:l_max for m in -l:l]
end

function rotate_test_R(R::Array{Complex{T}, 3}, rotation, l_max::Int) where {T<:AbstractFloat}
    """
    Rotate a test R tensor using the same coefficient convention as the crystal path.

    # Arguments:
    - R::Array{Complex{T},3}: Coefficients to rotate.
    - rotation: Proper rotor passed to the Wigner D construction.
    - l_max::Int: The largest angular mode.

    # Returns:
    - Array{Complex{T},3}: The rotated coefficients.
    """

    rotated = similar(R)
    blocks = SF.WignerRotations.wigner_d_blocks(rotation, l_max)
    for object in axes(R, 1), q_idx in axes(R, 2), l in 0:l_max
        block = (l * l + 1):((l + 1)^2)
        rotated[object, q_idx, block] .= blocks[l + 1] * R[object, q_idx, block]
    end
    return rotated
end

@testset "Stage timings" begin
    # Check that a timed stage returns its result and records one valid time.
    timings = Pair{String, Float64}[]
    value = SCarFFF.StageTimings.time_stage!(timings, "small stage") do
        sum(1:10)
    end
    @test value == 55
    @test length(timings) == 1
    @test timings[1].second >= 0

    # Check that the printed summary identifies both the run and the measured stage.
    output = mktemp() do _, io
        redirect_stdout(io) do
            SCarFFF.StageTimings.print_stage_timings("Test timings", timings)
        end
        seekstart(io)
        read(io, String)
    end
    @test occursin("Test timings", output)
    @test occursin("small stage", output)
    @test occursin('%', output)
end

function build_small_R_case(::Type{T}, tmp_dir::String) where {T<:AbstractFloat}
    """
    Construct the small molecular inputs used to test construct_R_tensor.

    # Arguments:
    - T::Type{<:AbstractFloat}: Floating-point precision used by the inputs.
    - tmp_dir::String: Temporary directory for the generated HDF5 files.

    # Returns:
    - inputs::Tuple: Positional arguments accepted by construct_R_tensor.
    """

    # Use the package's two-orbital mock molecule and the normal tensor-construction path.
    molecular_path = joinpath(tmp_dir, "mock_$(T).h5")
    SCarFFF.write_mock_molecular_data_file(molecular_path, T)
    mol = SF.ReadBasisSet.get_molecular_data(molecular_path; precision = T)

    # Construct the molecular tensors needed as input to construct_R_tensor.
    M_ij, sigma_ij, R_ij, R_ij_mod, R_ij_hat =
        SF.ConstructPairCoefficients.construct_pair_coefficients(mol)
    b_A, b_B, b_C = SF.ConstructbCoefficients.construct_b_coefficients(mol, R_ij)
    D_ij = SF.ConstructDTensor.construct_D_tensor(mol, M_ij, sigma_ij, b_A, b_B, b_C)

    # Use the package A tensor and generate the matching small Gaunt tensor for this test.
    data_dir = joinpath(dirname(pathof(SCarFFF)), "data")
    A_path = joinpath(data_dir, "A_tensors", T == Float32 ? "A_tensor_n2_f32.h5" : "A_tensor_n2_f64.h5")
    gaunt_path = joinpath(tmp_dir, "gaunt_$(T).h5")
    SF.PrecomputeGaunt.precompute_gaunt_coefficients(2, 4, 2, gaunt_path, T)
    W_ij = SF.ConstructWTensor.construct_W_tensor(D_ij, A_path; threshold = zero(T))

    # The asymmetric matrix and its transpose expose any accidental assumption that T_ij = T_ji.
    asymmetric = T[1 2; 3 4]
    density_matrices = Matrix{T}[mol.transition_matrices[1], asymmetric, Matrix(transpose(asymmetric))]
    q_grid = T[0, 0.25, 1]
    inputs = (W_ij, sigma_ij, R_ij_mod, R_ij_hat, q_grid, 2, gaunt_path,
              density_matrices, mol.cartesian_term_to_orbital)
    return (inputs = inputs, mol = mol, gaunt_path = gaunt_path)
end

@testset "Shared Cartesian-pair R contraction" begin
    mktempdir() do tmp_dir
        for T in (Float32, Float64)
            small_case = build_small_R_case(T, tmp_dir)
            inputs = small_case.inputs
            W_ij, sigma_ij, R_ij_mod, R_ij_hat, q_grid, l_max, gaunt_path,
                density_matrices, term_to_orbital = inputs

            # Computing all matrices together must agree with computing each one separately.
            batched = RT.construct_R_tensor(inputs...)
            single = cat((RT.construct_R_tensor(
                W_ij, sigma_ij, R_ij_mod, R_ij_hat, q_grid, l_max, gaunt_path,
                Matrix{T}[density], term_to_orbital) for density in density_matrices)...; dims = 1)

            tolerance = T == Float32 ? 2e-5 : 2e-12
            @test batched ≈ single rtol = tolerance atol = tolerance
            @test size(batched) == (length(density_matrices), length(q_grid), (l_max + 1)^2)

            # Check a previously known value, in case the single and batched implementations agree, but on the wrong value. 
            @test sum(abs2, batched) ≈ T(15822.173160627013) rtol = tolerance
            @test batched[1, 2, 1] ≈ Complex{T}(T(3.2200705644984984), 0) rtol = tolerance atol = tolerance
            @test batched[2, 3, 5] ≈ Complex{T}(T(-0.11710249835435621), 0) rtol = tolerance atol = tolerance
            @test batched[3, 2, 9] ≈ Complex{T}(T(-0.007409117682069378), 0) rtol = tolerance atol = tolerance

            # Check T_{μμ} on a Cartesian diagonal and T_{μν} + T_{νμ} otherwise. If two
            # distinct Cartesian terms belong to the same AO, the second case correctly gives 2T_{μμ}.
            full_bins, pair_i, pair_j, pair_density_weights =
                RT.build_pair_density_weights(W_ij, density_matrices, term_to_orbital)
            @test length(full_bins) == length(pair_i) == length(pair_j)
            for pair_idx in eachindex(pair_i)
                i = pair_i[pair_idx]
                j = pair_j[pair_idx]
                mu = term_to_orbital[i]
                nu = term_to_orbital[j]
                expected = i == j ? density_matrices[2][mu, nu] :
                           density_matrices[2][mu, nu] + density_matrices[2][nu, mu]
                @test pair_density_weights[2, pair_idx] == expected
            end

            # Check R_{ℓ,-m}(q) = (-1)^(m-ℓ) R_{ℓm}*(q) for every reconstructed coefficient.
            scale = maximum(abs, batched)
            for l in 0:l_max, m in 1:l
                positive_key = l * l + l + m + 1
                negative_key = l * l + l - m + 1
                reconstructed = ((-1)^(m - l)) .* conj.(view(batched, :, :, positive_key))
                @test view(batched, :, :, negative_key) ≈ reconstructed rtol = tolerance atol = tolerance * max(scale, one(T))
            end

            # At q = 0 all ℓ > 0 coefficients vanish, leaving only the monopole.
            for l in 1:l_max, m in -l:l
                key = l * l + l + m + 1
                @test maximum(abs, view(batched, :, 1, key)) <= tolerance * max(scale, one(T))
            end
        end
    end
end

@testset "Pair-frame convention" begin
    T = Float64
    z = SVector{3, T}(0, 0, 1)

    # S_inverse must carry the separation to z, including both axial edge cases.
    directions = (
        LinearAlgebra.normalize(SVector{3, T}(0.3, -0.4, 0.5)),
        SVector{3, T}(1, 0, 0),
        z,
        -z,
    )
    for direction in directions
        S_inverse = CJ.rotation_to_pair_frame(direction)
        matrix = SMatrix{3, 3, T, 9}(Quaternionic.to_rotation_matrix(S_inverse))
        @test matrix * direction ≈ z atol = 2e-14
    end

    # Check the complete identity used to rotate coefficients into pair coordinates:
    # Y_l^m(S q) = sum_mu D_mu,m(S_inverse) Y_l^mu(q).
    direction = LinearAlgebra.normalize(SVector{3, T}(0.21, -0.37, 0.91))
    q_pair = LinearAlgebra.normalize(SVector{3, T}(-0.43, 0.72, 0.31))
    S_inverse = CJ.rotation_to_pair_frame(direction)
    S = transpose(SMatrix{3, 3, T, 9}(Quaternionic.to_rotation_matrix(S_inverse)))
    blocks = SF.WignerRotations.wigner_d_blocks(S_inverse, 3)
    Y_global = harmonic_values(S * q_pair, 3)
    Y_pair = harmonic_values(q_pair, 3)

    for l in 0:3
        block = (l * l + 1):((l + 1)^2)
        source = Complex{T}[complex(0.13 * (m + 4), -0.07 * (m - 2)) for m in -l:l]
        pair_coefficients = blocks[l + 1] * source
        @test sum(source .* Y_global[block]) ≈
              sum(pair_coefficients .* Y_pair[block]) atol = 2e-13
    end
end

@testset "Ground, difference and nuclear densities" begin
    T = Float64

    # Check the AO formulas with non-square occupied and virtual blocks.
    C_o = Complex{T}[1 0; 0 1; 0.2 -0.1]
    C_v = Complex{T}[0.1; -0.3; 0.9;;]
    X = Complex{T}[0.4; -0.2;;]
    Y = Complex{T}[0.05; 0.12;;]
    ground = DM.build_ground_state_matrix(C_o)
    difference = DM.build_difference_matrix(X, Y, C_o, C_v)
    A = conj(X) * transpose(X) + Y * adjoint(Y)
    B = transpose(X) * conj(X) + adjoint(Y) * Y
    @test ground ≈ C_o * adjoint(C_o)
    @test difference ≈ C_v * B * adjoint(C_v) - C_o * A * adjoint(C_o)
    # Both are observable densities, so both must come out Hermitian.
    @test ground ≈ adjoint(ground)
    @test difference ≈ adjoint(difference)

    mktempdir() do tmp_dir
        small_case = build_small_R_case(T, tmp_dir)
        W_ij, sigma_ij, R_ij_mod, R_ij_hat, q_grid, l_max, gaunt_path, _, term_to_orbital =
            small_case.inputs
        mol = small_case.mol

        # Batch order is transitions, ground, then one difference density per transition.
        # As in, all_R = [transition_R_1, transition_R_2, ..., ground_R, difference_R_1, difference_R_2, ...].
        densities = Matrix{T}[
            mol.transition_matrices[1], mol.ground_state_matrix, mol.difference_matrices[1]]
        all_R = RT.construct_R_tensor(
            W_ij, sigma_ij, R_ij_mod, R_ij_hat, q_grid, l_max, gaunt_path,
            densities, term_to_orbital)
        transition_R, difference_R, Xi_R = DF.split_density_R_tensors(
            all_R, 1, mol.atom_coordinates, mol.nuclear_charges, q_grid, l_max)

        # Check that splitting the transitions didn't change them.
        @test transition_R == all_R[1:1, :, :]
        # Check that the charge densities are zero at the orgin, as these are just integrals over the molecule charge density, N_e - sum_I Z_I = 0.
        @test all(iszero, difference_R[:, 1, :])
        @test all(iszero, Xi_R[:, 1, :])

        # Expanding Z_lm against the harmonics must reproduce the point-charge sum it came from,
        # Σ_I Z_I exp(i q · R_I), which checks the i^l, Bessel and conj(Y) factors together.
        nuclear_R = DF.construct_nuclear_R(
            mol.atom_coordinates, mol.nuclear_charges, q_grid, 8)
        q_idx = 2
        q_invA = q_grid[q_idx] * SF.CrystalLattice.KEV_TO_INV_ANGSTROM
        q_hat = LinearAlgebra.normalize(SVector{3, T}(0.31, -0.52, 0.79))
        # Build the form factor from the harmonics and Z_lm coefficients.
        reconstructed = sum(vec(nuclear_R[1, q_idx, :]) .* harmonic_values(q_hat, 8))
        # Directly compute the form factor at this q value, sum_I Z_I exp(i q · R_I).
        direct = sum(T(Z) * cis(q_invA * dot(q_hat, SVector{3, T}(mol.atom_coordinates[I, :])))
                     for (I, Z) in enumerate(mol.nuclear_charges))
        # Check the two agree.
        @test reconstructed ≈ direct atol = 2e-13

        # A real density obeys R_l,-m = (-1)^(m-l) R_lm*.
        for l in 0:8, m in 1:l
            positive = l * l + l + m + 1
            negative = l * l + l - m + 1
            @test nuclear_R[1, :, negative] ≈
                  ((-1)^(m - l)) .* conj.(nuclear_R[1, :, positive]) atol = 2e-13
        end
    end
end

@testset "Hamiltonian diagonal correction" begin
    T = Float64
    identity_rotation = Quaternionic.rotor(T[1, 0, 0, 0])
    image = BH.CrystalImage{T}(1, identity_rotation, one(T), SVector{3, T}(0, 0, 0))
    # Build a fake molecule at the origin with two transitions at 2.1 and 2.8 eV.
    basis = BH.build_excitation_basis([image], [T[2.1, 2.8]])
    # Add a diagonal correction.
    BH.set_diagonal_corrections!(basis, T[-0.12, 0.07])
    eigensystem = BH.BlochEigensystem(basis)
    BH.build_bloch_hamiltonian!(eigensystem, basis, SVector{3, T}(0, 0, 0))
    # Check that D is on the diagonal as E + D: 2.1 - 0.12 and 2.8 + 0.07.
    @test diag(eigensystem.H) ≈ Complex{T}.(T[1.98, 2.87])

    # This is the state produced by --no-couplings: no J and D pinned to zero. Evaluated at a
    # general k, not k = 0, so that any surviving k dependence would show up.
    BH.set_diagonal_corrections!(basis, zeros(T, 2))
    BH.build_bloch_hamiltonian!(eigensystem, basis, SVector{3, T}(0.2, -0.1, 0.3))
    # Check that no diagonal correction was applied.
    @test eigensystem.H ≈ Diagonal(Complex{T}.(basis.energies))
    # Check that passing a single correction (there are two transitions, so need two), throws an error.
    @test_throws ErrorException BH.set_diagonal_corrections!(basis, T[1])
end

@testset "Shared short-range J and D pass" begin
    T = Float64
    l_max = 1
    q_grid = collect(range(zero(T), T(4), length = 41))
    n_q = length(q_grid)
    n_keys = (l_max + 1)^2

    # Build a fake R tensor for one density, obeying the negative-m relation of a real, neutral one.
    base = zeros(Complex{T}, 1, n_q, n_keys)
    for (q_idx, q) in enumerate(q_grid)
        radial = q * exp(-q * q / 2)
        base[1, q_idx, 1] = T(0.15) * q * radial
        base[1, q_idx, 2] = complex(T(0.3), T(-0.2)) * radial
        base[1, q_idx, 3] = T(0.11)im * radial
        base[1, q_idx, 4] = conj(base[1, q_idx, 2]) # (-1)^{m-l} * conj(base[1, q_idx, 2]). m - l = 0 here.
    end
    # Turn it into the three tensors the crystal path wants, for two molecules. The scalings differ
    # so that D_i←j and D_j←i are not equal, and the reverse-frame branch cannot pass by coincidence.
    transition_R = vcat(base, T(0.7) .* base)
    difference_R = vcat(T(0.4) .* base, T(-0.25) .* base)
    Xi_R = vcat(T(0.8) .* base, T(1.1) .* base)

    # Two unrotated molecules in one cell, with only the ΔR = 0 cell in the neighbour list. A
    # molecule does not interact with itself, so the sole job is the interaction between them.
    identity_rotation = Quaternionic.rotor(T[1, 0, 0, 0])
    images = [
        BH.CrystalImage{T}(1, identity_rotation, one(T), SVector{3, T}(-0.7, 0.1, 0.2)),
        BH.CrystalImage{T}(1, identity_rotation, one(T), SVector{3, T}(1.2, -0.4, 0.8)),
    ]
    basis = BH.build_excitation_basis(images, [T[2.0]])
    cells = CJ.NeighbourCells{T}(
        [SVector{3, T}(0, 0, 0)], [(0, 0, 0)], [1])

    mktempdir() do tmp_dir
        gaunt_path = joinpath(tmp_dir, "coupling_gaunt.h5")
        SF.PrecomputeGaunt.precompute_gaunt_coefficients(
            l_max, l_max, 2 * l_max, gaunt_path, T)

        # Folding D into the same pass must leave J exactly as the J-only route computed it.
        old_J = CJ.compute_couplings(
            transition_R, basis, cells, q_grid, l_max, gaunt_path)
        new_J, diagonal = CJ.compute_crystal_corrections(
            transition_R, difference_R, Xi_R, basis, cells, q_grid, l_max, gaunt_path)
        @test new_J.values ≈ old_J.values atol = 1e-13
        @test all(isfinite, diagonal)

        # With no change in the excited-state density, every directed D interaction vanishes.
        _, zero_diagonal = CJ.compute_crystal_corrections(
            transition_R, zero(difference_R), Xi_R, basis, cells, q_grid, l_max, gaunt_path)
        @test all(iszero, zero_diagonal)

        # A common rotation of both densities and their separation cannot alter a Coulomb integral.
        axis = LinearAlgebra.normalize(SVector{3, T}(0.2, -0.5, 0.7))
        angle = T(0.83)
        common_rotation = Quaternionic.rotor(T[
            cos(angle / 2), (sin(angle / 2) .* axis)...])
        rotation_matrix = SMatrix{3, 3, T, 9}(
            Quaternionic.to_rotation_matrix(common_rotation))
        rotated_images = [BH.CrystalImage{T}(
            1, identity_rotation, one(T), rotation_matrix * image.translation) for image in images]
        rotated_basis = BH.build_excitation_basis(rotated_images, [T[2.0]])
        rotated_transition = rotate_test_R(transition_R, common_rotation, l_max)
        rotated_difference = rotate_test_R(difference_R, common_rotation, l_max)
        rotated_Xi = rotate_test_R(Xi_R, common_rotation, l_max)
        rotated_J, rotated_diagonal = CJ.compute_crystal_corrections(
            rotated_transition, rotated_difference, rotated_Xi, rotated_basis, cells,
            q_grid, l_max, gaunt_path)
        @test rotated_J.values ≈ new_J.values rtol = 2e-12 atol = 2e-12
        @test rotated_diagonal ≈ diagonal rtol = 2e-12 atol = 2e-12

        # A molecule must feel neighbours on both sides, so one molecule repeated periodically
        # should be shifted twice as much as one with a single neighbour.
        shift = SVector{3, T}(2.1, -0.3, 0.4)

        # One neighbour: two identical molecules a distance shift apart, in a single cell.
        pair_images = [
            BH.CrystalImage{T}(1, identity_rotation, one(T), SVector{3, T}(0, 0, 0)),
            BH.CrystalImage{T}(1, identity_rotation, one(T), shift),
        ]
        pair_basis = BH.build_excitation_basis(pair_images, [T[2.0]])
        identical_transition = vcat(base, base)
        identical_difference = vcat(T(0.4) .* base, T(0.4) .* base)
        identical_Xi = vcat(T(0.8) .* base, T(0.8) .* base)
        _, directed_pair = CJ.compute_crystal_corrections(
            identical_transition, identical_difference, identical_Xi, pair_basis, cells,
            q_grid, l_max, gaunt_path)

        # Two neighbours: the same molecule alone, repeated at ±shift. The zero cell is itself.
        single_basis = BH.build_excitation_basis(images[1:1], [T[2.0]])
        lattice_cells = CJ.NeighbourCells{T}(
            [-shift, SVector{3, T}(0, 0, 0), shift],
            [(-1, 0, 0), (0, 0, 0), (1, 0, 0)], [3, 2, 1])
        _, repeated_image = CJ.compute_crystal_corrections(
            base, T(0.4) .* base, T(0.8) .* base,
            single_basis, lattice_cells, q_grid, l_max, gaunt_path)
        # Left and right agree only because ΔN and Ξ share a shape here. For a lopsided pair of
        # densities they do not, and the two directions differ even in sign.
        @test only(repeated_image) ≈ 2 * directed_pair[1] rtol = 2e-12 atol = 2e-12

        # A molecule does not shift its own energy, so with no neighbours there is nothing to feel.
        _, self_only = CJ.compute_crystal_corrections(
            base, T(0.4) .* base, T(0.8) .* base,
            single_basis, cells, q_grid, l_max, gaunt_path)
        @test all(iszero, self_only)
    end
end

@testset "Ewald diagonal correction" begin
    T = Float64
    l_max = 1
    q_grid = collect(range(zero(T), T(6), length = 241))
    n_q = length(q_grid)
    density = zeros(Complex{T}, 1, n_q, 4)
    for (q_idx, q) in enumerate(q_grid)
        # A neutral spherical density gives an absolutely convergent reference sum, without
        # mixing the Ewald implementation test with the separate dipole-boundary convention.
        density[1, q_idx, 1] = T(0.8) * q * q * exp(-q * q / 2)
    end
    transition_R = copy(density)
    difference_R = T(0.6) .* density
    Xi_R = T(-0.9) .* density

    identity_rotation = Quaternionic.rotor(T[1, 0, 0, 0])
    image = BH.CrystalImage{T}(1, identity_rotation, one(T), SVector{3, T}(0, 0, 0))
    basis = BH.build_excitation_basis([image], [T[2.0]])
    lattice = SF.CrystalLattice.build_crystal_lattice(T[4 0 0; 0 4 0; 0 0 4], T)

    mktempdir() do tmp_dir
        gaunt_path = joinpath(tmp_dir, "ewald_gaunt.h5")
        SF.PrecomputeGaunt.precompute_gaunt_coefficients(
            l_max, l_max, 2 * l_max, gaunt_path, T)

        # A large spherical direct sum provides the reference boundary condition.
        direct_cells = CJ.enumerate_neighbour_cells(lattice, T(12))
        _, direct_D = CJ.compute_crystal_corrections(
            transition_R, difference_R, Xi_R, basis, direct_cells,
            q_grid, l_max, gaunt_path)

        # η splits the sum between the two halves and must cancel out of the total.
        ewald_values = T[]
        for eta in T[0.4, 0.5]
            parameters = SF.Ewald.choose_ewald_parameters(
                lattice, T(1e-8); eta = eta)
            short_cells = CJ.enumerate_neighbour_cells(lattice, parameters.R_max)
            long_range = SF.Ewald.build_ewald_long_range(
                transition_R, [image.translation], basis.image_of, lattice,
                q_grid, l_max, parameters)
            _, short_D = CJ.compute_crystal_corrections(
                transition_R, difference_R, Xi_R, basis, short_cells,
                q_grid, l_max, gaunt_path; parameters = parameters)
            long_D = SF.Ewald.compute_ewald_diagonal(
                difference_R, Xi_R, q_grid, l_max, long_range)
            push!(ewald_values, only(short_D .+ long_D))
        end

        # Loose tolerances because of the reference, not the split. The pair interaction is dead by
        # 8 Å, but each pair carries a numerical floor of order 1e-5, so summing more cells adds
        # noise faster than signal. 12 Å is about as good as the direct sum gets.
        @test ewald_values[1] ≈ ewald_values[2] rtol = 1.5e-2 atol = 2e-5
        @test ewald_values[1] ≈ only(direct_D) rtol = 1.5e-2 atol = 2e-5
    end
end
