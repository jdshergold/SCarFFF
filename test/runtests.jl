using Test
using SCarFFF
using LinearAlgebra
using Quaternionic
using SphericalHarmonics
using StaticArrays
using HDF5

const SF = SCarFFF.SphericalFormFactor
const RT = SF.ConstructRTensor
const DF = SF.DensityFormFactors
const CJ = SF.CouplingJ
const BH = SF.BlochHamiltonian
const DM = SF.ReadBasisSet.DensityMatrices
const RB = SF.ReadBasisSet

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

function neutral_test_R(n_densities::Int, q_grid::Vector{T}, l_max::Int)::Array{Complex{T}, 3} where {T<:AbstractFloat}
    """
    Construct neutral test densities with different coefficients for each row and angular mode.

    # Arguments:
    - n_densities::Int: The number of densities.
    - q_grid::Vector{T}: The |q| grid, in inverse Angstroms.
    - l_max::Int: The largest angular mode.

    # Returns:
    - Array{Complex{T},3}: The test R tensor, with dimensions (n_densities, n_q, n_keys).
    """

    R = zeros(Complex{T}, n_densities, length(q_grid), (l_max + 1)^2)
    for density in 1:n_densities, (q_idx, q) in enumerate(q_grid), l in 0:l_max
        radial = (l == 0 ? q^2 : q^l) * exp(-q^2 / 2)
        R[density, q_idx, l * l + l + 1] =
            (isodd(l) ? Complex{T}(0, 1) : Complex{T}(1, 0)) *
            T(0.03 + 0.04 * density) * radial
        for m in 1:l
            positive_key = l * l + l + m + 1
            negative_key = l * l + l - m + 1
            value = Complex{T}(T(0.07 * density + 0.02 * m), T(-0.03 * density + 0.05 * l)) * radial
            R[density, q_idx, positive_key] = value
            R[density, q_idx, negative_key] = T((-1)^(m - l)) * conj(value)
        end
    end
    return R
end

function angular_D_reference(
        Delta_R::Array{Complex{T}, 3},
        Xi_R::Matrix{Complex{T}},
        separation::SVector{3, T},
        q_grid::Vector{T},
        radial_weights::Vector{T},
    )::Vector{Complex{T}} where {T<:AbstractFloat}
    """
    Integrate Δ(q) Ξ*(q) exp(i q ⋅ d) over directions without pair frames or Gaunt coefficients.

    # Arguments:
    - Delta_R::Array{Complex{T},3}: The Δ coefficients, one row per pair of states.
    - Xi_R::Matrix{Complex{T}}: The neighbouring molecule's Ξ coefficients, indexed by q and key.
    - separation::SVector{3,T}: The molecular separation, in Angstroms.
    - q_grid::Vector{T}: The |q| grid, in inverse Angstroms.
    - radial_weights::Vector{T}: The radial quadrature weights, including any Ewald kernel.

    # Returns:
    - Vector{Complex{T}}: One contribution to D per pair of states, in eV.
    """

    n_pairs, n_q, n_keys = size(Delta_R)
    l_max = isqrt(n_keys) - 1

    # Gauss-Legendre integrates in cos(θ), so its weights already include sin(θ) dθ.
    n_theta = 24
    n_phi = 48
    legendre = eigen(SymTridiagonal(zeros(T, n_theta),
        T[i / sqrt(4 * i^2 - 1) for i in 1:(n_theta - 1)]))
    cos_theta = legendre.values
    theta_weights = T(2) .* abs2.(legendre.vectors[1, :])
    phi_weight = T(2 * π / n_phi)
    Delta_matrix = reshape(Delta_R, n_pairs * n_q, n_keys)
    contribution = zeros(Complex{T}, n_pairs)

    for theta_idx in 1:n_theta, phi_idx in 1:n_phi
        phi = T(2 * π * (phi_idx - 1) / n_phi)
        sin_theta = sqrt(one(T) - cos_theta[theta_idx]^2)
        direction = SVector{3, T}(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta[theta_idx])
        harmonics = harmonic_values(direction, l_max)
        Delta_values = reshape(Delta_matrix * harmonics, n_pairs, n_q)
        Xi_values = Xi_R * harmonics
        angular_weight = theta_weights[theta_idx] * phi_weight
        for q_idx in 1:n_q
            weight = angular_weight * radial_weights[q_idx] *
                     cis(q_grid[q_idx] * dot(direction, separation)) * conj(Xi_values[q_idx])
            for pair in 1:n_pairs
                contribution[pair] += weight * Delta_values[pair, q_idx]
            end
        end
    end

    prefactor = T(SF.CrystalLattice.ALPHA_EM * SF.CrystalLattice.HBAR_C_EV_ANGSTROM / (2 * π^2))
    return prefactor .* contribution
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

    # Generate the small angular tensors in the test directory.
    A_path = joinpath(tmp_dir, "A_$(T).h5")
    SF.PrecomputeATensor.precompute_A_tensor(2, A_path, T)
    gaunt_path = joinpath(tmp_dir, "gaunt_$(T).h5")
    SF.PrecomputeGaunt.precompute_gaunt_coefficients(2, 4, 2, gaunt_path, T)
    W_ij = SF.ConstructWTensor.construct_W_tensor(D_ij, A_path; threshold = zero(T))

    # The asymmetric matrix and its transpose expose any accidental assumption that T_ij = T_ji.
    asymmetric = T[1 2; 3 4]
    density_matrices = Matrix{T}[mol.T_gs[1], asymmetric, Matrix(transpose(asymmetric))]
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
            @test sum(abs2, batched) ≈ T(15822.167346872453) rtol = tolerance
            @test batched[1, 2, 1] ≈ Complex{T}(T(3.220068689449471), 0) rtol = tolerance atol = tolerance
            @test batched[2, 3, 5] ≈ Complex{T}(T(-0.11710258766693253), 0) rtol = tolerance atol = tolerance
            @test batched[3, 2, 9] ≈ Complex{T}(T(-0.007409123348153007), 0) rtol = tolerance atol = tolerance

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
    ground = DM.build_T_gg(C_o)
    difference = DM.build_Delta_T(X, Y, X, Y, C_o, C_v)
    A = conj(X) * transpose(X) + Y * adjoint(Y)
    B = transpose(X) * conj(X) + adjoint(Y) * Y
    @test ground ≈ C_o * adjoint(C_o)
    @test difference ≈ C_v * B * adjoint(C_v) - C_o * A * adjoint(C_o)
    # Both are observable densities, so both must come out Hermitian.
    @test ground ≈ adjoint(ground)
    @test difference ≈ adjoint(difference)

    # A second state, so the off-diagonal pair s != t can be built. The amplitudes are complex and
    # unrelated to the first state's.
    X_t = Complex{T}[0.15 + 0.3im; 0.45;;]
    Y_t = Complex{T}[-0.08; 0.02 - 0.11im;;]
    Delta_st = DM.build_Delta_T(X, Y, X_t, Y_t, C_o, C_v)
    Delta_ts = DM.build_Delta_T(X_t, Y_t, X, Y, C_o, C_v)
    # A^(ts) and B^(ts), written out here independently of the function being tested.
    A_ts = conj(X) * transpose(X_t) + Y_t * adjoint(Y)
    B_ts = transpose(X_t) * conj(X) + adjoint(Y) * Y_t
    @test Delta_st ≈ C_v * B_ts * adjoint(C_v) - C_o * A_ts * adjoint(C_o)
    # Swapping the states gives the adjoint, since A^(ts)† = A^(st).
    @test Delta_st ≈ adjoint(Delta_ts)
    # Check that they really do differ.
    @test !isapprox(Delta_st, Delta_ts)

    # Real amplitudes, which is what RKS gives and all that to_real_density lets through. Being
    # adjoints is not enough on its own: for a real density the adjoint is just the transpose.
    X_real = T[0.22; 0.37;;]
    Y_real = T[-0.06; 0.09;;]
    C_o_real = real(C_o)
    C_v_real = real(C_v)
    real_st = DM.build_Delta_T(real(X), real(Y), X_real, Y_real, C_o_real, C_v_real)
    real_ts = DM.build_Delta_T(X_real, Y_real, real(X), real(Y), C_o_real, C_v_real)
    @test real_st ≈ transpose(real_ts)
    # Neither matrix is symmetric by itself, so the test below is not comparing a density to itself.
    @test !isapprox(real_st, real_ts)

    # Swapping s and t transposes Δ in its AO indices μν. The density is built from φ_μ(r) φ_ν(r),
    # which is symmetric in μν because the AOs are real, so the antisymmetric part of Δ contributes
    # nothing and the two orderings give the same coefficients.
    mktempdir() do tmp_dir
        small_case = build_small_R_case(T, tmp_dir)
        W_ij, sigma_ij, R_ij_mod, R_ij_hat, q_grid, l_max, gaunt_path, _, term_to_orbital =
            small_case.inputs
        both = RT.construct_R_tensor(
            W_ij, sigma_ij, R_ij_mod, R_ij_hat, q_grid, l_max, gaunt_path,
            Matrix{T}[real_st, real_ts], term_to_orbital)
        @test both[1, :, :] ≈ both[2, :, :]
    end

    # The (s, t) pair tables.
    for n_transitions in (1, 2, 5)
        pairs = DM.build_transition_pairs(n_transitions)
        n_pairs = div(n_transitions * (n_transitions + 1), 2)
        @test length(pairs.transitions_of_pair) == n_pairs
        # Every pair is stored with s <= t, and the reverse map is symmetric and inverts it.
        for pair in 1:n_pairs
            (s, t) = pairs.transitions_of_pair[pair]
            @test s <= t
            @test pairs.pair_of_transitions[s, t] == pair
            @test pairs.pair_of_transitions[t, s] == pair
        end
        # The diagonal pairs sit at the triangular numbers.
        for t in 1:n_transitions
            @test pairs.pair_of_transitions[t, t] == div(t * (t + 1), 2)
        end
    end
    # One transition leaves one pair, which is the layout D had before it carried pairs.
    @test DM.build_transition_pairs(1).transitions_of_pair == [(1, 1)]
    @test_throws ErrorException DM.build_transition_pairs(0)

    mktempdir() do tmp_dir
        small_case = build_small_R_case(T, tmp_dir)
        W_ij, sigma_ij, R_ij_mod, R_ij_hat, q_grid, l_max, gaunt_path, _, term_to_orbital =
            small_case.inputs
        mol = small_case.mol

        # Batch order is transitions, ground, then one Δ density per (s, t) pair.
        # As in, all_R = [transition_R_1, transition_R_2, ..., ground_R, Delta_R_1, Delta_R_2, ...].
        densities = Matrix{T}[mol.T_gs[1], mol.T_gg]
        append!(densities, RB.build_Delta_T_matrices(mol, [1]))
        all_R = RT.construct_R_tensor(
            W_ij, sigma_ij, R_ij_mod, R_ij_hat, q_grid, l_max, gaunt_path,
            densities, term_to_orbital)
        transition_R, Delta_R, Xi_R = DF.split_density_R_tensors(
            all_R, 1, mol.atom_coordinates, mol.nuclear_charges, q_grid, l_max)

        # Check that splitting the transitions didn't change them.
        @test transition_R == all_R[1:1, :, :]
        # Check that the charge densities are zero at the orgin, as these are just integrals over the molecule charge density, N_e - sum_I Z_I = 0.
        @test all(iszero, Delta_R[:, 1, :])
        @test all(iszero, Xi_R[:, 1, :])

        # Two transitions give three pairs, so the batch is 2 + 1 + 3 and Δ starts after the ground state.
        # This molecule only has one state, so invent a second one by scaling its amplitudes. Any Δ
        # built from X and Y is neutral at q = 0, so the split's neutrality check still applies.
        X_1 = mol.X_states[1]
        Y_1 = mol.Y_states[1]
        X_2 = T(0.7) .* X_1
        Y_2 = T(-0.4) .* Y_1
        Delta_pair_matrices = Matrix{T}[
            DM.build_Delta_T(X_1, Y_1, X_1, Y_1, mol.C_o, mol.C_v),
            DM.build_Delta_T(X_1, Y_1, X_2, Y_2, mol.C_o, mol.C_v),
            DM.build_Delta_T(X_2, Y_2, X_2, Y_2, mol.C_o, mol.C_v)]
        two_state_densities = Matrix{T}[mol.T_gs[1], mol.T_gs[1],
                                        mol.T_gg]
        append!(two_state_densities, Delta_pair_matrices)
        two_state_R = RT.construct_R_tensor(
            W_ij, sigma_ij, R_ij_mod, R_ij_hat, q_grid, l_max, gaunt_path,
            two_state_densities, term_to_orbital)
        _, two_state_Delta, _ = DF.split_density_R_tensors(
            two_state_R, 2, mol.atom_coordinates, mol.nuclear_charges, q_grid, l_max)
        # Check that we have the right number of pairs, and that they pop up in the right place in R.
        @test size(two_state_Delta, 1) == 3
        for pair in 1:3
            @test two_state_Delta[pair, 2:end, :] == two_state_R[3 + pair, 2:end, :]
        end
        # The off-diagonal pair is a different density from either diagonal one.
        @test !isapprox(two_state_Delta[2, :, :], two_state_Delta[1, :, :])
        @test !isapprox(two_state_Delta[2, :, :], two_state_Delta[3, :, :])

        # Check that the Δ matrices come out in the right order.
        selected_mol = deepcopy(mol)
        X_3 = T(-0.3) .* X_1
        Y_3 = T(0.8) .* Y_1
        append!(selected_mol.X_states, [X_2, X_3])
        append!(selected_mol.Y_states, [Y_2, Y_3])
        selected_Delta = RB.build_Delta_T_matrices(selected_mol, [3, 1])
        @test selected_Delta[1] ≈ real(DM.build_Delta_T(
            X_3, Y_3, X_3, Y_3, mol.C_o, mol.C_v))
        @test selected_Delta[2] ≈ real(DM.build_Delta_T(
            X_3, Y_3, X_1, Y_1, mol.C_o, mol.C_v))
        @test selected_Delta[3] ≈ real(DM.build_Delta_T(
            X_1, Y_1, X_1, Y_1, mol.C_o, mol.C_v))

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

    T = Float64
    l_max = 2
    q_grid = T[0, 0.3, 0.8]
    rotation = Quaternionic.rotor(T[cos(0.4), 0, sin(0.4), 0])
    images = [
        BH.CrystalImage{T}(2, rotation, -one(T), SVector{3, T}(0, 0, 0)),
        BH.CrystalImage{T}(1, rotation, one(T), SVector{3, T}(2, 1, -0.5)),
    ]
    basis = BH.build_excitation_basis(images, [T[2.1, 2.8], T[2.2, 3.0]])
    conformer_R = [neutral_test_R(2, q_grid, l_max), T(0.7) .* neutral_test_R(2, q_grid, l_max)]
    conformer_Delta = [neutral_test_R(3, q_grid, l_max), T(-0.4) .* neutral_test_R(3, q_grid, l_max)]
    conformer_Xi = [neutral_test_R(1, q_grid, l_max), T(1.2) .* neutral_test_R(1, q_grid, l_max)]
    rotated_R, rotated_Delta, rotated_Xi = SF.rotate_crystal_R_tensors(
        conformer_R, conformer_Delta, conformer_Xi, images, basis, l_max)
    @test size(rotated_Delta) == (6, length(q_grid), (l_max + 1)^2)

    # Check that the rotated coefficients match the expected rotation of the conformer coefficients, including the det(S)^l factor.
    for (image_idx, image) in enumerate(images)
        for (conformers, rotated, n_densities) in
                ((conformer_R, rotated_R, 2), (conformer_Delta, rotated_Delta, 3), (conformer_Xi, rotated_Xi, 1))
            expected = rotate_test_R(conformers[image.conformer_index], image.R_tilde, l_max)
            for l in 0:l_max
                keys = (l * l + 1):((l + 1)^2)
                expected[:, :, keys] .*= image.det_rotation^l
            end
            rows = ((image_idx - 1) * n_densities + 1):(image_idx * n_densities)
            @test rotated[rows, :, :] ≈ expected rtol = 2e-13 atol = 2e-13
        end
    end
end

@testset "Hamiltonian D correction" begin
    T = Float64
    identity_rotation = Quaternionic.rotor(T[1, 0, 0, 0])
    image = BH.CrystalImage{T}(1, identity_rotation, one(T), SVector{3, T}(0, 0, 0))
    # Build a fake molecule at the origin with two transitions at 2.1 and 2.8 eV.
    basis = BH.build_excitation_basis([image], [T[2.1, 2.8]])
    # Add a diagonal correction.
    BH.set_D_couplings!(basis, reshape(T[-0.12, 0, 0, 0.07], 1, 2, 2))
    eigensystem = BH.BlochEigensystem(basis)
    BH.build_bloch_hamiltonian!(eigensystem, basis, SVector{3, T}(0, 0, 0))
    # Check that D is on the diagonal as E + D: 2.1 - 0.12 and 2.8 + 0.07.
    @test diag(eigensystem.H) ≈ Complex{T}.(T[1.98, 2.87])

    # This is the state produced by --no-couplings: no J and D pinned to zero. Evaluated at a
    # general k, not k = 0, so that any surviving k dependence would show up.
    BH.set_D_couplings!(basis, zeros(T, 1, 2, 2))
    BH.build_bloch_hamiltonian!(eigensystem, basis, SVector{3, T}(0.2, -0.1, 0.3))
    # Check that no diagonal correction was applied.
    @test eigensystem.H ≈ Diagonal(Complex{T}.(basis.Delta_E_monomer))
    # Check that passing a block for one transition, when there are two, throws an error.
    @test_throws ErrorException BH.set_D_couplings!(basis, ones(T, 1, 1, 1))

    for T in (Float32, Float64)
        identity_rotation = Quaternionic.rotor(T[1, 0, 0, 0])
        images = [
            BH.CrystalImage{T}(1, identity_rotation, one(T), SVector{3, T}(0, 0, 0)),
            BH.CrystalImage{T}(2, identity_rotation, one(T), SVector{3, T}(2, -1, 0.5)),
        ]
        basis = BH.build_excitation_basis(images, [T[2.1, 2.8], T[2.2, 3.0]])
        D_blocks = zeros(T, 2, 2, 2)
        D_blocks[1, :, :] = T[-0.12 0.18; 0.18 0.07]
        D_blocks[2, :, :] = T[0.03 -0.09; -0.09 -0.04]
        BH.set_D_couplings!(basis, D_blocks)

        # Each molecule has its own block. The zeros between them enforce δ_{A_i B_j}.
        expected_D = T[-0.12 0.18 0 0; 0.18 0.07 0 0;
                       0 0 0.03 -0.09; 0 0 -0.09 -0.04]
        expected_H = Diagonal(basis.Delta_E_monomer) + expected_D
        @test basis.D_couplings == expected_D
        eigensystem = BH.BlochEigensystem(basis)
        tolerance = T == Float32 ? 2e-6 : 2e-13

        # D mixes transitions on the same molecule even with J absent, and has no k dependence.
        for k in (SVector{3, T}(0, 0, 0), SVector{3, T}(0.2, -0.1, 0.3))
            BH.solve_bloch_hamiltonian!(eigensystem, basis, k)
            @test eigensystem.H ≈ expected_H atol = tolerance
            @test eigensystem.Delta_E ≈ eigvals(Symmetric(expected_H)) atol = tolerance
            @test expected_H * eigensystem.C ≈
                  eigensystem.C * Diagonal(eigensystem.Delta_E) atol = tolerance
        end

        # Invalid blocks must be rejected before replacing the previous D.
        @test_throws ErrorException BH.set_D_couplings!(basis, zeros(T, 1, 2, 2))
        for value in (T(NaN), T(Inf))
            invalid_D = copy(D_blocks)
            invalid_D[1, 1, 1] = value
            @test_throws ErrorException BH.set_D_couplings!(basis, invalid_D)
        end
        asymmetric_D = copy(D_blocks)
        asymmetric_D[1, 2, 1] += one(T)
        @test_throws ErrorException BH.set_D_couplings!(basis, asymmetric_D)
        @test basis.D_couplings == expected_D

        # Replacing D with zero must clear the off-diagonal entries as well as the shifts.
        BH.set_D_couplings!(basis, zero(D_blocks))
        BH.build_bloch_hamiltonian!(eigensystem, basis, SVector{3, T}(0, 0, 0))
        @test eigensystem.H == Diagonal(Complex{T}.(basis.Delta_E_monomer))
    end
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
    Delta_R = vcat(T(0.4) .* base, T(-0.25) .* base)
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
            l_max, 2 * l_max, l_max, gaunt_path, T)

        # Folding D into the same pass must leave J exactly as the J-only route computed it.
        old_J = CJ.compute_couplings(
            transition_R, basis, cells, q_grid, l_max, gaunt_path)
        new_J, D_blocks = CJ.compute_crystal_corrections(
            transition_R, Delta_R, Xi_R, basis, cells, q_grid, l_max, gaunt_path)
        @test new_J.J_deltaR ≈ old_J.J_deltaR atol = 1e-13
        @test all(isfinite, D_blocks)

        # With no change in the excited-state density, every directed D interaction vanishes.
        _, zero_D = CJ.compute_crystal_corrections(
            transition_R, zero(Delta_R), Xi_R, basis, cells, q_grid, l_max, gaunt_path)
        @test all(iszero, zero_D)

        # A common rotation of both densities and their separation cannot alter a Coulomb integral.
        axis = LinearAlgebra.normalize(SVector{3, T}(0.2, -0.5, 0.7))
        angle = T(0.83)
        common_rotation = Quaternionic.rotor(T[
            cos(angle / 2), (sin(angle / 2) .* axis)...])
        rotation_matrix = SMatrix{3, 3, T, 9}(
            Quaternionic.to_rotation_matrix(common_rotation))
        rotated_images = [BH.CrystalImage{T}(
            1, identity_rotation, one(T), rotation_matrix * image.tau) for image in images]
        rotated_basis = BH.build_excitation_basis(rotated_images, [T[2.0]])
        rotated_transition = rotate_test_R(transition_R, common_rotation, l_max)
        rotated_Delta = rotate_test_R(Delta_R, common_rotation, l_max)
        rotated_Xi = rotate_test_R(Xi_R, common_rotation, l_max)
        rotated_J, rotated_D = CJ.compute_crystal_corrections(
            rotated_transition, rotated_Delta, rotated_Xi, rotated_basis, cells,
            q_grid, l_max, gaunt_path)
        @test rotated_J.J_deltaR ≈ new_J.J_deltaR rtol = 2e-12 atol = 2e-12
        @test rotated_D ≈ D_blocks rtol = 2e-12 atol = 2e-12

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
        identical_Delta = vcat(T(0.4) .* base, T(0.4) .* base)
        identical_Xi = vcat(T(0.8) .* base, T(0.8) .* base)
        _, directed_pair = CJ.compute_crystal_corrections(
            identical_transition, identical_Delta, identical_Xi, pair_basis, cells,
            q_grid, l_max, gaunt_path)

        # Two neighbours: the same molecule alone, repeated at ±shift. The zero cell is itself.
        single_basis = BH.build_excitation_basis(images[1:1], [T[2.0]])
        lattice_cells = CJ.NeighbourCells{T}(
            [-shift, SVector{3, T}(0, 0, 0), shift],
            [(-1, 0, 0), (0, 0, 0), (1, 0, 0)], [3, 2, 1])
        _, repeated_image = CJ.compute_crystal_corrections(
            base, T(0.4) .* base, T(0.8) .* base,
            single_basis, lattice_cells, q_grid, l_max, gaunt_path)
        # Left and right agree only because Δ and Ξ share a shape here. For a lopsided pair of
        # densities they do not, and the two directions differ even in sign.
        @test only(repeated_image) ≈ 2 * directed_pair[1] rtol = 2e-12 atol = 2e-12

        # A molecule does not shift its own energy, so with no neighbours there is nothing to feel.
        _, self_only = CJ.compute_crystal_corrections(
            base, T(0.4) .* base, T(0.8) .* base,
            single_basis, cells, q_grid, l_max, gaunt_path)
        @test all(iszero, self_only)
    end

    T = Float64
    l_max = 2
    q_grid = collect(range(zero(T), T(6), length = 241))
    identity_rotation = Quaternionic.rotor(T[1, 0, 0, 0])
    images = [
        BH.CrystalImage{T}(1, identity_rotation, one(T), SVector{3, T}(-0.7, 0.1, 0.2)),
        BH.CrystalImage{T}(2, identity_rotation, one(T), SVector{3, T}(1.2, -0.4, 0.8)),
    ]
    basis = BH.build_excitation_basis(images, [T[2.1, 2.8], T[2.2, 3.0]])
    cells = CJ.NeighbourCells{T}([SVector{3, T}(0, 0, 0)], [(0, 0, 0)], [1])
    pairs = DM.build_transition_pairs(2)
    Delta_R = neutral_test_R(6, q_grid, l_max)
    Xi_R = neutral_test_R(2, q_grid, l_max)
    transition_R = vcat(Delta_R[1:2, :, :], Delta_R[4:5, :, :])
    lattice = SF.build_crystal_lattice(T[8 0 0; 0 8 0; 0 0 8], T)
    parameters = SF.choose_ewald_parameters(lattice, 2, T(1e-8), SF.D_TRUNCATION; eta = T(0.4))

    mktempdir() do tmp_dir
        gaunt_path = joinpath(tmp_dir, "off_diagonal_D_gaunt.h5")
        SF.PrecomputeGaunt.precompute_gaunt_coefficients(l_max, 2 * l_max, l_max, gaunt_path, T)

        # Compare both kernels against a direct angular integral. This checks the rotation of the
        # potential back to Δ, including complex Wigner matrices and the reversed separation.
        for split in (nothing, parameters)
            couplings, D_blocks = CJ.compute_crystal_corrections(
                transition_R, Delta_R, Xi_R, basis, cells, q_grid, l_max, gaunt_path;
                J_parameters = split, D_parameters = split)
            J_only = CJ.compute_couplings(
                transition_R, basis, cells, q_grid, l_max, gaunt_path; parameters = split)
            @test couplings.J_deltaR == J_only.J_deltaR
            @test size(D_blocks) == (2, 2, 2)
            @test all(abs.(D_blocks[:, 1, 2]) .> 1e-5)
            @test D_blocks[:, 1, 2] == D_blocks[:, 2, 1]

            radial_weights = CJ.quadrature_weights(q_grid, false)
            if split !== nothing
                radial_weights .*= one(T) .- exp.(-(q_grid ./ (2 * split.eta)).^2)
            end
            for image in 1:2
                neighbour = 3 - image
                rows = ((image - 1) * 3 + 1):(image * 3)
                separation = images[image].tau - images[neighbour].tau
                reference = angular_D_reference(
                    Delta_R[rows, :, :], Xi_R[neighbour, :, :], separation, q_grid, radial_weights)
                @test maximum(abs, imag(reference)) < 1e-12
                for (pair, (s, t)) in enumerate(pairs.transitions_of_pair)
                    @test D_blocks[image, s, t] ≈ real(reference[pair]) rtol = 2e-11 atol = 2e-12
                end
            end

            # Rotating every density and the separation together cannot change D.
            axis = LinearAlgebra.normalize(SVector{3, T}(0.2, -0.5, 0.7))
            angle = T(0.83)
            rotation = Quaternionic.rotor(T[cos(angle / 2), (sin(angle / 2) .* axis)...])
            rotation_matrix = SMatrix{3, 3, T, 9}(Quaternionic.to_rotation_matrix(rotation))
            rotated_images = [BH.CrystalImage{T}(
                image.conformer_index, identity_rotation, one(T), rotation_matrix * image.tau)
                for image in images]
            rotated_basis = BH.build_excitation_basis(rotated_images, [T[2.1, 2.8], T[2.2, 3.0]])
            _, rotated_D = CJ.compute_crystal_corrections(
                rotate_test_R(transition_R, rotation, l_max), rotate_test_R(Delta_R, rotation, l_max),
                rotate_test_R(Xi_R, rotation, l_max), rotated_basis, cells, q_grid, l_max, gaunt_path;
                J_parameters = split, D_parameters = split)
            @test rotated_D ≈ D_blocks rtol = 2e-12 atol = 2e-12
        end

        # With either charge density absent, no element of D may survive.
        for (Delta, Xi) in ((zero(Delta_R), Xi_R), (Delta_R, zero(Xi_R)))
            _, zero_D = CJ.compute_crystal_corrections(
                transition_R, Delta, Xi, basis, cells, q_grid, l_max, gaunt_path)
            @test all(iszero, zero_D)
        end
    end
end

@testset "J cell list trimming" begin
    # Ensure that J doesn't do extra work in light of R_max being longer for D than J.
    T = Float64
    l_max = 1
    q_grid = collect(range(zero(T), T(4), length = 41))
    identity_rotation = Quaternionic.rotor(T[1, 0, 0, 0])
    images = [
        BH.CrystalImage{T}(1, identity_rotation, one(T), SVector{3, T}(-0.7, 0.1, 0.2)),
        BH.CrystalImage{T}(1, identity_rotation, one(T), SVector{3, T}(1.2, -0.4, 0.8)),
    ]
    basis = BH.build_excitation_basis(images, [T[2.0, 2.6]])
    Delta_R = neutral_test_R(6, q_grid, l_max)
    Xi_R = neutral_test_R(2, q_grid, l_max)
    transition_R = vcat(Delta_R[1:2, :, :], Delta_R[4:5, :, :])
    lattice = SF.build_crystal_lattice(T[5 0 0; 0 5.5 0; 0 0 6], T)

    # J stops at 7 Å while D runs to 13, so the cell list is sized for D and most of it is dead
    # weight for J.
    J_split = SF.Ewald.EwaldParameters{T}(T(0.45), T(7), T(3), T(1e-3))
    D_split = SF.Ewald.EwaldParameters{T}(T(0.45), T(13), T(3), T(1e-3))
    wide_cells = CJ.enumerate_neighbour_cells(lattice, T(13) + CJ.image_translation_span(basis))
    narrow_cells = CJ.enumerate_neighbour_cells(lattice, T(7) + CJ.image_translation_span(basis))
    @test length(wide_cells.Delta_R) > length(narrow_cells.Delta_R)

    mktempdir() do tmp_dir
        gaunt_path = joinpath(tmp_dir, "trim_gaunt.h5")
        SF.PrecomputeGaunt.precompute_gaunt_coefficients(l_max, 2 * l_max, l_max, gaunt_path, T)

        trimmed, _ = CJ.compute_crystal_corrections(
            transition_R, Delta_R, Xi_R, basis, wide_cells, q_grid, l_max, gaunt_path;
            J_parameters = J_split, D_parameters = D_split)
        reference, _ = CJ.compute_crystal_corrections(
            transition_R, Delta_R, Xi_R, basis, narrow_cells, q_grid, l_max, gaunt_path;
            J_parameters = J_split, D_parameters = D_split)

        # The trim must result in a shorter list.
        @test length(trimmed.Delta_R) < length(wide_cells.Delta_R)
        @test !all(iszero, trimmed.J_deltaR)

        # Every cell kept must carry something, bar ΔR = 0 which is kept for the self term.
        for cell in eachindex(trimmed.Delta_R)
            iszero(trimmed.Delta_R[cell]) && continue
            @test !all(iszero, view(trimmed.J_deltaR, cell, :, :))
        end

        # The two cell lists are in different orders, so compare the Bloch sum J(k) over ΔR, which
        # does not depend on the order, rather than the stored arrays.
        bloch(couplings, k) = sum(couplings.J_deltaR[cell, :, :] *
                                  cis(dot(k, couplings.Delta_R[cell]))
                                  for cell in eachindex(couplings.Delta_R))
        for fraction in (SVector{3, T}(0, 0, 0), SVector{3, T}(0.25, 0, 0),
                         SVector{3, T}(0.1, -0.3, 0.2), SVector{3, T}(0.125, 0.125, 0.125))
            k = SVector{3, T}(transpose(lattice.reciprocal) * fraction)
            @test bloch(trimmed, k) ≈ bloch(reference, k) rtol = 1e-12 atol = 1e-14
        end
    end
end

@testset "Ewald cutoff choice" begin
    T = Float64
    # Base these tests on the unit cell for benzene.
    lattice = SF.build_crystal_lattice(T[7.46 0 0; 0 9.66 0; 0 0 7.03], T)
    Z_mol = 4

    for truncation in (SF.D_TRUNCATION, SF.J_TRUNCATION)
        R_0 = truncation.k_R * cbrt(lattice.V_cell / Z_mol)
        for epsilon_tot in T[1e-2, 1e-3, 1e-6]
            split = SF.choose_ewald_parameters(lattice, Z_mol, epsilon_tot, truncation)
            # Both cutoffs must deliver exactly the error they were asked for, which is the whole
            # point of carrying the prefactors and the offset.
            eta_eff = inv(sqrt(inv(split.eta^2) + inv(truncation.sigma_0^2)))
            delta_R = truncation.A_R * exp(-split.eta^2 * (split.R_max - R_0)^2)
            delta_G = truncation.A_G * exp(-split.G_max^2 / (4 * eta_eff^2))
            @test delta_R ≈ epsilon_tot / sqrt(T(2)) rtol = 1e-6
            @test delta_G ≈ epsilon_tot / sqrt(T(2)) rtol = 1e-6
            # Check that the cutoffs are sensible.
            @test split.R_max > R_0
            @test split.G_max > 0

            # A fixed η must satisfy the same two laws, just without minimising the work.
            fixed = SF.choose_ewald_parameters(
                lattice, Z_mol, epsilon_tot, truncation; eta = T(0.35))
            @test fixed.eta == T(0.35)
            fixed_eta_eff = inv(sqrt(inv(fixed.eta^2) + inv(truncation.sigma_0^2)))
            @test truncation.A_R * exp(-fixed.eta^2 * (fixed.R_max - R_0)^2) ≈
                  epsilon_tot / sqrt(T(2)) rtol = 1e-6
            @test truncation.A_G * exp(-fixed.G_max^2 / (4 * fixed_eta_eff^2)) ≈
                  epsilon_tot / sqrt(T(2)) rtol = 1e-6
        end
    end

    # Set q_max to half the optimal G_max. G_max should be capped at q_max, and R_max extended to
    # keep the requested error.
    for truncation in (SF.D_TRUNCATION, SF.J_TRUNCATION)
        R_0 = truncation.k_R * cbrt(lattice.V_cell / Z_mol)
        free = SF.choose_ewald_parameters(lattice, Z_mol, T(1e-3), truncation)
        capped = SF.choose_ewald_parameters(
            lattice, Z_mol, T(1e-3), truncation; q_reach = free.G_max / 2)
        @test capped.G_max ≈ free.G_max / 2
        @test capped.R_max > free.R_max
        @test capped.eta < free.eta
        eta_eff = inv(sqrt(inv(capped.eta^2) + inv(truncation.sigma_0^2)))
        @test truncation.A_R * exp(-capped.eta^2 * (capped.R_max - R_0)^2) ≈
              T(1e-3) / sqrt(T(2)) rtol = 1e-6
        @test truncation.A_G * exp(-capped.G_max^2 / (4 * eta_eff^2)) ≈
              T(1e-3) / sqrt(T(2)) rtol = 1e-6

        # If q_max is beyond the optimal G_max, nothing changes.
        roomy = SF.choose_ewald_parameters(
            lattice, Z_mol, T(1e-3), truncation; q_reach = 2 * free.G_max)
        @test roomy.G_max == free.G_max
        @test roomy.R_max == free.R_max
    end

    # D converges more slowly in both sums, so it must end up with the larger cutoff.
    D_split = SF.choose_ewald_parameters(lattice, Z_mol, T(1e-3), SF.D_TRUNCATION)
    J_split = SF.choose_ewald_parameters(lattice, Z_mol, T(1e-3), SF.J_TRUNCATION)
    @test D_split.R_max > J_split.R_max
    @test D_split.G_max > J_split.G_max

    # σ_0 only narrows the reciprocal cutoff, since it leads to a faster decay.
    no_decay = SF.EwaldTruncation(
        SF.J_TRUNCATION.A_R, SF.J_TRUNCATION.A_G, Inf, SF.J_TRUNCATION.k_R)
    wider = SF.choose_ewald_parameters(lattice, Z_mol, T(1e-3), no_decay; eta = J_split.eta)
    @test wider.G_max > J_split.G_max
    @test wider.R_max ≈ J_split.R_max

    # Tighter tolerances lead to larger cutoffs.
    @test SF.choose_ewald_parameters(lattice, Z_mol, T(1e-6), SF.D_TRUNCATION).R_max >
          SF.choose_ewald_parameters(lattice, Z_mol, T(1e-3), SF.D_TRUNCATION).R_max

    # An error looser than the laws can express is decreased to the cap.
    for truncation in (SF.D_TRUNCATION, SF.J_TRUNCATION)
        loose = SF.choose_ewald_parameters(lattice, Z_mol, T(0.5), truncation)
        capped = SF.choose_ewald_parameters(lattice, Z_mol, T(SF.Ewald.MAX_EPSILON_TOT), truncation)
        @test loose.epsilon_tot == T(SF.Ewald.MAX_EPSILON_TOT)
        @test loose.eta == capped.eta
        @test loose.R_max == capped.R_max
        @test loose.G_max == capped.G_max
        # The cap has to be sensible, or else it might as well not exist.
        @test loose.R_max > truncation.k_R * cbrt(lattice.V_cell / Z_mol)
        @test loose.G_max > 0
    end

    @test_throws ErrorException SF.choose_ewald_parameters(
        lattice, Z_mol, T(1.5), SF.D_TRUNCATION)
    @test_throws ErrorException SF.choose_ewald_parameters(
        lattice, 0, T(1e-3), SF.D_TRUNCATION)
    @test_throws ErrorException SF.choose_ewald_parameters(
        lattice, Z_mol, T(1e-3), SF.D_TRUNCATION; cost_ratio = 0)
end

@testset "Ewald D correction" begin
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
    transition_R = vcat(density, T(0.7) .* density)
    Delta_R = vcat(T(0.6) .* density, T(-0.35) .* density, T(0.2) .* density)
    Xi_R = T(-0.9) .* density

    # The dipole of the Q = 0 term is read off a single small q.
    q_bc = T(1e-4)
    density_bc = zeros(Complex{T}, 1, 1, 4)
    density_bc[1, 1, 1] = T(0.8) * q_bc * q_bc * exp(-q_bc * q_bc / 2)
    transition_R_bc = vcat(density_bc, T(0.7) .* density_bc)
    Delta_R_bc = vcat(T(0.6) .* density_bc, T(-0.35) .* density_bc, T(0.2) .* density_bc)
    Xi_R_bc = T(-0.9) .* density_bc

    identity_rotation = Quaternionic.rotor(T[1, 0, 0, 0])
    image = BH.CrystalImage{T}(1, identity_rotation, one(T), SVector{3, T}(0, 0, 0))
    basis = BH.build_excitation_basis([image], [T[2.0, 2.5]])
    lattice = SF.CrystalLattice.build_crystal_lattice(T[4 0 0; 0 4 0; 0 0 4], T)

    mktempdir() do tmp_dir
        gaunt_path = joinpath(tmp_dir, "ewald_gaunt.h5")
        SF.PrecomputeGaunt.precompute_gaunt_coefficients(
            l_max, 2 * l_max, l_max, gaunt_path, T)

        # A large spherical direct sum provides the reference boundary condition.
        direct_cells = CJ.enumerate_neighbour_cells(lattice, T(12))
        _, direct_D = CJ.compute_crystal_corrections(
            transition_R, Delta_R, Xi_R, basis, direct_cells,
            q_grid, l_max, gaunt_path)

        @test size(direct_D) == (1, 2, 2)
        @test abs(direct_D[1, 1, 2]) > 1e-6

        # η splits the sum between the two halves and must cancel out of the total. The two splits must not
        # depend on one another.
        ewald_values = Array{T, 3}[]
        J_parameters = SF.Ewald.EwaldParameters{T}(T(0.45), T(9), T(4), T(1e-8))
        for eta in T[0.4, 0.5]
            D_parameters = SF.Ewald.EwaldParameters{T}(eta, T(11), 2 * eta * sqrt(T(18.76)), T(1e-8))
            short_cells = CJ.enumerate_neighbour_cells(
                lattice, max(J_parameters.R_max, D_parameters.R_max))
            long_range = SF.Ewald.build_ewald_long_range(
                transition_R, transition_R_bc, [image.tau, image.tau], basis.image_of, lattice,
                q_grid, q_bc, l_max, J_parameters)
            _, short_D = CJ.compute_crystal_corrections(
                transition_R, Delta_R, Xi_R, basis, short_cells,
                q_grid, l_max, gaunt_path;
                J_parameters = J_parameters, D_parameters = D_parameters)
            long_D = SF.Ewald.compute_ewald_D(
                Delta_R, Xi_R, Delta_R_bc, Xi_R_bc,
                q_grid, q_bc, basis.n_transitions, l_max, long_range, D_parameters)
            push!(ewald_values, short_D .+ long_D)
            @test size(long_D) == (1, 2, 2)
            @test long_D[1, 1, 2] == long_D[1, 2, 1]
        end

        # Loose tolerances because of the reference, not the split. The pair interaction is dead by
        # 8 Å, but each pair carries a numerical floor of order 1e-5, so summing more cells adds
        # noise faster than signal. 12 Å is about as good as the direct sum gets.
        @test ewald_values[1] ≈ ewald_values[2] rtol = 1.5e-2 atol = 2e-5
        @test ewald_values[1] ≈ direct_D rtol = 1.5e-2 atol = 2e-5
    end

    T = Float64
    l_max = 2
    q_grid = collect(range(zero(T), T(6), length = 241))
    q_bc = T(1e-4)
    identity_rotation = Quaternionic.rotor(T[1, 0, 0, 0])
    images = [
        BH.CrystalImage{T}(1, identity_rotation, one(T), SVector{3, T}(-0.7, 0.1, 0.2)),
        BH.CrystalImage{T}(1, identity_rotation, one(T), SVector{3, T}(1.2, -0.4, 0.8)),
    ]
    basis = BH.build_excitation_basis(images, [T[2.1, 2.8]])
    Delta_R = neutral_test_R(6, q_grid, l_max)
    Xi_R = neutral_test_R(2, q_grid, l_max)
    Delta_bc = neutral_test_R(6, T[q_bc], l_max)
    Xi_bc = neutral_test_R(2, T[q_bc], l_max)
    transition_R = vcat(Delta_R[1:2, :, :], Delta_R[4:5, :, :])
    transition_bc = vcat(Delta_bc[1:2, :, :], Delta_bc[4:5, :, :])
    tau_of_lambda_i = [images[image].tau for image in basis.image_of]
    lattice = SF.build_crystal_lattice(T[8 0 0; 0 8 0; 0 0 8], T)
    parameters = SF.choose_ewald_parameters(lattice, 2, T(1e-8), SF.D_TRUNCATION; eta = T(0.4))
    pairs = DM.build_transition_pairs(2)

    vacuum = SF.build_ewald_long_range(
        transition_R, transition_bc, tau_of_lambda_i, basis.image_of, lattice,
        q_grid, q_bc, l_max, parameters; epsilon_b = one(T))
    conductor = SF.build_ewald_long_range(
        transition_R, transition_bc, tau_of_lambda_i, basis.image_of, lattice,
        q_grid, q_bc, l_max, parameters; epsilon_b = T(Inf))
    dielectric = SF.build_ewald_long_range(
        transition_R, transition_bc, tau_of_lambda_i, basis.image_of, lattice,
        q_grid, q_bc, l_max, parameters; epsilon_b = T(4))
    D_vacuum = SF.compute_ewald_D(Delta_R, Xi_R, Delta_bc, Xi_bc,
        q_grid, q_bc, basis.n_transitions, l_max, vacuum, parameters)
    D_conductor = SF.compute_ewald_D(Delta_R, Xi_R, Delta_bc, Xi_bc,
        q_grid, q_bc, basis.n_transitions, l_max, conductor, parameters)
    D_dielectric = SF.compute_ewald_D(Delta_R, Xi_R, Delta_bc, Xi_bc,
        q_grid, q_bc, basis.n_transitions, l_max, dielectric, parameters)

    # The boundary term is the scalar product of the Δ and total Ξ dipoles, including s != t.
    boundary = zeros(T, 2, 2, 2)
    for image in 1:2, (pair, (s, t)) in enumerate(pairs.transitions_of_pair)
        row = (image - 1) * 3 + pair
        dipole_product = real(sum(Delta_bc[row, 1, key] * conj(sum(Xi_bc[:, 1, key]))
                                 for key in 2:4)) / q_bc^2
        boundary[image, s, t] = vacuum.prefactor * dipole_product / (4 * π)
        boundary[image, t, s] = boundary[image, s, t]
    end
    @test all(abs.(boundary[:, 1, 2]) .> 1e-5)
    @test D_vacuum - D_conductor ≈ boundary rtol = 1e-11 atol = 1e-12
    @test D_dielectric - D_conductor ≈ boundary / 3 rtol = 1e-11 atol = 1e-12

    # Use the analytic test densities at G directly, so the reference does not reuse the Hermite
    # tables. The tolerance allows for interpolation on the finite q grid.
    reciprocal = zeros(Complex{T}, 2, 2, 2)
    for G in vacuum.G_vectors
        G_norm = norm(G)
        (G_norm < 1e-10 || G_norm > parameters.G_max) && continue
        harmonics = harmonic_values(G / G_norm, l_max)
        Delta_at_G = neutral_test_R(6, T[G_norm], l_max)
        Xi_at_G = neutral_test_R(2, T[G_norm], l_max)
        total_Xi = sum(cis(dot(G, images[image].tau)) *
                       sum(Xi_at_G[image, 1, :] .* harmonics) for image in 1:2)
        weight = vacuum.prefactor * exp(-(G_norm / (2 * parameters.eta))^2) / G_norm^2
        for image in 1:2, (pair, (s, t)) in enumerate(pairs.transitions_of_pair)
            row = (image - 1) * 3 + pair
            value = weight * cis(dot(G, images[image].tau)) *
                    sum(Delta_at_G[row, 1, :] .* harmonics) * conj(total_Xi)
            reciprocal[image, s, t] += value
            s == t && continue
            reciprocal[image, t, s] += conj(value)
        end
    end
    @test maximum(abs, imag(reciprocal)) < 1e-12

    # At zero separation the angular integral reduces to an inner product. Only the Ξ on the
    # same molecule is removed, rather than the whole unit-cell charge density.
    q_step = q_grid[2] - q_grid[1]
    self_weights = q_step .* exp.(-(q_grid ./ (2 * parameters.eta)).^2)
    self_weights[1] /= 2
    self_weights[end] /= 2
    self_D = zeros(T, 2, 2, 2)
    self_prefactor = T(SF.CrystalLattice.ALPHA_EM * SF.CrystalLattice.HBAR_C_EV_ANGSTROM / (2 * π^2))
    for image in 1:2, (pair, (s, t)) in enumerate(pairs.transitions_of_pair)
        row = (image - 1) * 3 + pair
        value = self_prefactor * real(sum(self_weights[q_idx] * Delta_R[row, q_idx, key] *
            conj(Xi_R[image, q_idx, key]) for q_idx in eachindex(q_grid), key in 1:(l_max + 1)^2))
        self_D[image, s, t] = value
        self_D[image, t, s] = value
    end
    @test D_conductor + self_D ≈ real(reciprocal) rtol = 2e-5 atol = 2e-7

    # Cutting the reciprocal sum below the shortest reciprocal vector, with conducting boundaries,
    # leaves exactly -D_self.
    no_reciprocal = SF.Ewald.EwaldParameters{T}(
        parameters.eta, parameters.R_max, T(0.1), parameters.epsilon_tot)
    self_only = SF.compute_ewald_D(Delta_R, Xi_R, Delta_bc, Xi_bc,
        q_grid, q_bc, basis.n_transitions, l_max, conductor, no_reciprocal)
    @test self_only ≈ -self_D rtol = 2e-12 atol = 2e-12

    # Malformed pair and boundary tensors must fail before the contraction starts.
    @test_throws ErrorException SF.compute_ewald_D(Delta_R[1:4, :, :], Xi_R, Delta_bc, Xi_bc,
        q_grid, q_bc, basis.n_transitions, l_max, vacuum, parameters)
    @test_throws ErrorException SF.compute_ewald_D(Delta_R, Xi_R, Delta_bc[1:4, :, :], Xi_bc,
        q_grid, q_bc, basis.n_transitions, l_max, vacuum, parameters)
    @test_throws ErrorException SF.compute_ewald_D(Delta_R, Xi_R, Delta_bc, Xi_bc,
        q_grid, q_bc, 1, l_max, vacuum, parameters)
end

@testset "Structure function energy axis" begin
    # Check that the grid spans from 5 eV to 8 eV, with a σ_E = 3 dE buffer on either end.
    grid = SF.build_energy_grid(5.0, 8.0, 0.01)

    @test grid.energy_step ≈ 0.01
    @test grid.sigma ≈ 3 * grid.energy_step
    @test grid.half_window == 3

    # Check that there are either 5 or 6 points (for σ_E = 3 ΔE)
    for offset in range(0, grid.energy_step, length = 9)[1:8]
        buffer = zeros(Float64, length(grid.E), 1, 1, 1)
        occupancy = zeros(Bool, length(grid.E), 1)
        SF.accumulate_delta!(buffer, occupancy, grid, 6.5 + offset, 1.0, 1, 1, 1)
        @test count(!iszero, buffer) in (5, 6)
    end
    @test first(grid.E) ≈ 5.0 - grid.sigma
    @test last(grid.E) >= 8.0 + grid.sigma
    @test all(diff(grid.E) .≈ 0.01)

    # σ_E is where the mollifier reaches zero, with nothing at all beyond it. Check this.
    @test SF.kernel_shape(grid, grid.sigma) == 0.0
    @test SF.kernel_shape(grid, 1.01 * grid.sigma) == 0.0
    @test SF.kernel_shape(grid, 0.99 * grid.sigma) > 0.0

    # Check that halving the grid spacing halves σ_E.
    finer = SF.build_energy_grid(5.0, 8.0, 0.005)

    @test finer.sigma ≈ grid.sigma / 2
    @test length(finer.E) > length(grid.E)

    # Reject dE that are too small, as we would need too many grid points.
    @test_throws ErrorException SF.build_energy_grid(5.0, 8.0, 1e-7)
    @test_throws ErrorException SF.build_energy_grid(5.0, 8.0, -0.01)

    # Check that the delta approximation preserves the normalisation.
    for energy in (5.0, 6.5, 8.0, first(grid.E), last(grid.E))
        buffer = zeros(Float64, length(grid.E), 1, 1, 1)
        occupancy = zeros(Bool, length(grid.E), 1)
        SF.accumulate_delta!(buffer, occupancy, grid, energy, 3.25, 1, 1, 1)
        @test sum(buffer) * grid.energy_step ≈ 3.25
    end

    # Check everything in the structure factor is zero away from where the bands are.
    buffer = zeros(Float64, length(grid.E), 1, 1, 1)
    occupancy = zeros(Bool, length(grid.E), 1)
    SF.accumulate_delta!(buffer, occupancy, grid, 6.5, 1.0, 1, 1, 1)
    outside = abs.(grid.E .- 6.5) .> grid.sigma

    @test all(buffer[outside, 1, 1, 1] .== 0.0)

    # Also check that it nonzero where it should be.
    buffer = zeros(Float64, length(grid.E), 1, 1, 1)
    occupancy = zeros(Bool, length(grid.E), 1)
    SF.accumulate_delta!(buffer, occupancy, grid, 6.5, 1.0, 1, 1, 1)
    weights = buffer[:, 1, 1, 1]
    mean_energy = sum(weights .* grid.E) / sum(weights)

    @test mean_energy ≈ 6.5 atol = 1e-6
    @test grid.E[argmax(weights)] ≈ 6.5 atol = grid.energy_step

    # Check that the variance is what we expect.
    variance = sum(weights .* (grid.E .- mean_energy).^2) / sum(weights)
    @test sqrt(variance) ≈ 0.3976350541 * grid.sigma rtol = 5e-2

    # Check that multiple states are actually accumulated, and don't overwrite one another.
    buffer = zeros(Float64, length(grid.E), 1, 1, 1)
    occupancy = zeros(Bool, length(grid.E), 1)
    for (energy, magnitude) in ((5.5, 1.0), (6.0, 2.0), (7.25, 0.5))
        SF.accumulate_delta!(buffer, occupancy, grid, energy, magnitude, 1, 1, 1)
    end
    @test sum(buffer) * grid.energy_step ≈ 3.5

    # Make sure that all of the bins with stuff in are marked as occupied.
    @test all(buffer[.!occupancy[:, 1], 1, 1, 1] .== 0.0)
    @test count(occupancy) < length(grid.E)

    # The incoherent route spreads fixed transition energies, and integrating E must give back the
    # coefficients it started from.
    f_sq_lm = reshape(collect(1.0:12.0), 2, 2, 3)
    energies = [5.5, 7.0]
    structure_factor = SF.incoherent_intensity_lm_E(f_sq_lm, energies, grid)
    integral = SF.integrate_energy(structure_factor, grid)

    @test size(structure_factor) == (length(grid.E), 2, 3)
    @test integral ≈ dropdims(sum(f_sq_lm, dims = 1), dims = 1)

    # The saved structure function includes 2π/V_cell, with no state or molecule average.
    V_cell = 125.0
    SF.apply_structure_function_prefactor!(structure_factor, V_cell)
    @test SF.integrate_energy(structure_factor, grid) ≈
          (2π / V_cell) .* dropdims(sum(f_sq_lm, dims = 1), dims = 1)
    @test_throws ErrorException SF.apply_structure_function_prefactor!(copy(structure_factor), 0.0)
end

@testset "Energy axis trimming" begin
    # The axis is built wider than the bands need, so the empty ends come off before anything is
    # saved to disk.
    grid = SF.build_energy_grid(5.0, 8.0, 0.01)
    n_energy = length(grid.E)
    structure_factor = zeros(Float64, n_energy, 2, 3)

    # Put some nonzero physics in two bins.
    low_bin, high_bin = 20, n_energy - 15
    structure_factor[low_bin, 1, 1] = 1.0
    structure_factor[high_bin, 2, 3] = 2.0

    # Trim the grid.
    trimmed, trimmed_grid = SF.trim_energy_axis(structure_factor, grid)

    # Check the trimmed size is correct.
    @test size(trimmed) == (high_bin - low_bin + 1, 2, 3)
    @test length(trimmed_grid.E) == high_bin - low_bin + 1
    @test trimmed_grid.E[1] ≈ grid.E[low_bin]
    @test trimmed_grid.E[end] ≈ grid.E[high_bin]

    # Check that the trimming didn't drop anything nonzero.
    @test trimmed_grid.sigma == grid.sigma
    @test trimmed_grid.energy_step == grid.energy_step
    @test sum(trimmed) == sum(structure_factor)

    # Check that there are the right number of gaps in the middle.
    @test count(iszero, trimmed) == length(trimmed) - 2

    # Nothing to trim leaves the arrays alone rather than copying them.
    full = ones(Float64, n_energy, 2, 3)
    untrimmed, untrimmed_grid = SF.trim_energy_axis(full, grid)

    @test untrimmed === full
    @test untrimmed_grid === grid

    # An empty structure function is returned as it stands rather than trimmed to nothing.
    empty_factor = zeros(Float64, n_energy, 2, 3)
    unchanged, unchanged_grid = SF.trim_energy_axis(empty_factor, grid)

    @test unchanged === empty_factor
    @test length(unchanged_grid.E) == n_energy
end

function two_molecule_cell(::Type{T}, offset::SVector{3, T},
                           conformers::Vector{Int}) where {T<:AbstractFloat}
    """
    Build a unit cell holding two molecules that are unrelated by symmetry, each with a real image.

    Molecule two is molecule one translated by offset, which is the arrangement that makes the pair
    look interchangeable.

    # Arguments:
    - T::Type: The floating point type.
    - offset::SVector{3, T}: Where the second molecule sits relative to the first, in Å.
    - conformers::Vector{Int}: The conformer label of each of the four images.

    # Returns:
    - CrystalLatticeData{T}: The lattice.
    - Vector{CrystalImage{T}}: The four images.
    """

    # Construct the lattice and images.
    lattice = SF.build_crystal_lattice(T[12 0 -4; 0 6 0; 0 0 14], T)
    upright = Quaternionic.rotor(T[1, 0, 0, 0]) # Identity quaternion.
    turned = Quaternionic.rotor(T[0, 0, 1, 0]) # 180° rotation about y.
    first_place = SVector{3, T}(0, 3, 0) # Position of the first molecule.
    partner_place = SVector{3, T}(0, 0, 7) # Position of its symmetry partner.
    places = [first_place, partner_place, first_place + offset, partner_place + offset]
    rotations = [upright, turned, upright, turned]
    parities = T[1, -1, 1, -1] # Turns the rotation into a mirror op. 
    images = [SF.CrystalImage{T}(conformers[i], rotations[i], parities[i], places[i])
              for i in 1:4]
    return lattice, images
end

@testset "Symmetry with two independent molecules" begin
    T = Float64

    # Check that an offset that does not line up with the lattice does not produce any symops.
    runaway_offset = SVector{3, T}(12 * sqrt(2) / 4, 0, 0)
    lattice, images = two_molecule_cell(T, runaway_offset, [1, 1, 2, 2])
    operations = SF.derive_symmetry_operations(images, lattice)

    # Only 1 and 2, and 3 and 4 should be partners.
    @test length(operations) == 4
    for operation in operations
        @test operation.image_of in ([1, 2, 3, 4], [2, 1, 4, 3])
    end

    # Now an offset of exactly half the a vector. This closes the lattice, but the conformer
    # labels are different, so they shouldn't be registered as partners.
    half_cell = SVector{3, T}(6, 0, -2)
    lattice, images = two_molecule_cell(T, half_cell, [1, 1, 2, 2])
    operations = SF.derive_symmetry_operations(images, lattice)

    # Check that each operation maps a molecule to its partner, and that the conformer labels match.
    for operation in operations, image in 1:length(images)
        @test images[image].conformer_index ==
              images[operation.image_of[image]].conformer_index
    end

    # Now we say they're all the same conformer, so it should find that all are images of one another.
    _, merged = two_molecule_cell(T, half_cell, [1, 1, 1, 1])
    merged_operations = SF.derive_symmetry_operations(merged, lattice)

    @test length(merged_operations) > length(operations)
    @test any(operation.image_of[1] > 2 for operation in merged_operations)
end

function dipole_couplings(basis, images, lattice, reference_dipoles::Vector{Vector{SVector{3, T}}},
                          ) where {T<:AbstractFloat}
    """
    Build point dipole couplings J_{λλ'}(ΔR) for a cell, so that H(k) has off-diagonal structure.

    # Arguments:
    - basis: The localised excitation basis.
    - images: The images in the unit cell.
    - lattice: The crystal lattice.
    - reference_dipoles::Vector{Vector{SVector{3,T}}}: A dipole per conformer, per transition.

    # Returns:
    - CrystalCouplings{T}: The couplings, over the 3x3x3 block of neighbouring cells.
    """

    # Go up to 10 Å, which is enough to include the nearest neighbour of each molecule in the cell.
    cutoff = T(10)
    cell_vectors = SVector{3, T}[]
    # Build the cell vectors for the 5^3 block.
    for n1 in -2:2, n2 in -2:2, n3 in -2:2
        push!(cell_vectors, SVector{3, T}(n1 * lattice.direct[1, :] + n2 * lattice.direct[2, :] +
                                          n3 * lattice.direct[3, :]))
    end

    n_lambda_i = length(basis.Delta_E_monomer)
    dipoles = Vector{SVector{3, T}}(undef, n_lambda_i)
    for lambda in 1:n_lambda_i
        image = images[basis.image_of[lambda]]
        operation = SF.CrystalSymmetry.cartesian_operation(image)
        # Compute the dipoles for each image, by rotating hte reference dipoles.
        dipoles[lambda] = operation *
            reference_dipoles[image.conformer_index][basis.transition_of[lambda]]
    end

    values = zeros(Complex{T}, length(cell_vectors), n_lambda_i, n_lambda_i)
    for (cell_idx, cell) in enumerate(cell_vectors),
        lambda_j in 1:n_lambda_i, lambda in 1:n_lambda_i

        separation = images[basis.image_of[lambda]].tau -
                     images[basis.image_of[lambda_j]].tau - cell
        distance = norm(separation)
        # A molecule does not couple to itself in its own cell, and nothing couples past the cutoff.
        (distance < 1e-8 || distance > cutoff) && continue
        direction = separation / distance
        first_dipole = dipoles[lambda]
        second_dipole = dipoles[lambda_j]
        # Set J_{λλ'}(ΔR) = (d_λ · d_λ') / |distance|^3.
        values[cell_idx, lambda, lambda_j] = Complex{T}(
            (dot(first_dipole, second_dipole) -
             3 * dot(first_dipole, direction) * dot(second_dipole, direction)) / distance^3)
    end

    return BH.CrystalCouplings{T}(values, cell_vectors)
end

@testset "Symmetry relation on H(k)" begin
    """
    The reduction solves H(k) once per star and reconstructs the rest from
    
        H_{Λ(a) Λ(b)}(k_Λ) = H_{ab}(k) exp[-i k_Λ · (L_a - L_b)],    k_Λ = R_Λ k,

    so we check that this holds.
    """

    T = Float64

    # The same invented Z' = 2 cell as the testset above. Two molecules with one image each.
    lattice, images = two_molecule_cell(T, SVector{3, T}(6, 0, -2), [1, 1, 2, 2])

    # The two different molecules have excitations differing by 20 meV.
    basis = BH.build_excitation_basis(images, [T[4.10, 4.65], T[4.12, 4.67]])
    operations = SF.derive_symmetry_operations(images, lattice)
    n_lambda_i = length(basis.Delta_E_monomer)

    # The environment mixes transitions differently for the two conformers, but symmetry-related
    # images of the same conformer must carry the same D block.
    D_blocks = zeros(T, length(images), 2, 2)
    for (image_idx, image) in enumerate(images)
        D_blocks[image_idx, :, :] = image.conformer_index == 1 ?
            T[-0.12 0.08; 0.08 0.03] : T[0.04 -0.06; -0.06 -0.09]
    end
    BH.set_D_couplings!(basis, D_blocks)

    # Invent some reference dipoles, one for each transition of each conformer.
    reference_dipoles = [
        SVector{3, T}[SVector{3, T}(0.9, 0.2, -0.3), SVector{3, T}(-0.1, 0.7, 0.5)],
        SVector{3, T}[SVector{3, T}(0.4, -0.8, 0.1), SVector{3, T}(0.6, 0.3, 0.7)],
    ]
    couplings = dipole_couplings(basis, images, lattice, reference_dipoles)

    # Create the eigensystems at k and its symmetry partner, k_Λ.
    at_k = BH.BlochEigensystem(basis)
    at_k_lambda = BH.BlochEigensystem(basis)

    # Check the Hamiltonian behaves as intented.
    # First, that it has off diagonal terms at some reference k.
    BH.build_bloch_hamiltonian!(at_k, basis, SVector{3, T}(0.13, -0.07, 0.21), couplings)
    reference = Matrix(Hermitian(copy(at_k.H), :U))
    @test maximum(abs, reference - Diagonal(diag(reference))) > 1e-6
    # Now check that it has different values at a different k.
    BH.build_bloch_hamiltonian!(at_k, basis, SVector{3, T}(-0.31, 0.05, 0.11), couplings)
    @test maximum(abs, Matrix(Hermitian(copy(at_k.H), :U)) - reference) > 1e-6

    # And now construct it at a few different k.
    for k in (SVector{3, T}(0.13, -0.07, 0.21), SVector{3, T}(-0.31, 0.05, 0.11),
              SVector{3, T}(0.02, 0.19, -0.28))
        BH.build_bloch_hamiltonian!(at_k, basis, k, couplings)
        H = Matrix(Hermitian(copy(at_k.H), :U))

        # And at each of its symmetry partners.
        for operation in operations
            # k_Λ = R_Λ k, and time reversal flips the sign on top of that.
            reversal = operation.conjugate ? -one(T) : one(T)
            k_lambda = reversal * (operation.R_Lambda * k)
            BH.build_bloch_hamiltonian!(at_k_lambda, basis, k_lambda, couplings)
            H_lambda = Matrix(Hermitian(copy(at_k_lambda.H), :U))

            for a in 1:n_lambda_i, b in 1:n_lambda_i
                # Find the λ that the operation sends a and b to. λ is a pair (image, transition),
                # and only the image is permuted. The && chain assigns when both parts match.
                mapped_a = 0
                mapped_b = 0
                for lambda in 1:n_lambda_i
                    basis.transition_of[lambda] == basis.transition_of[a] &&
                        basis.image_of[lambda] == operation.image_of[basis.image_of[a]] &&
                        (mapped_a = lambda)
                    basis.transition_of[lambda] == basis.transition_of[b] &&
                        basis.image_of[lambda] == operation.image_of[basis.image_of[b]] &&
                        (mapped_b = lambda)
                end

                # Compute the phase factor from the translation.
                offset_a = operation.L_offsets[basis.image_of[a]]
                offset_b = operation.L_offsets[basis.image_of[b]]
                phase = cis(-dot(k_lambda, offset_a - offset_b))
                # Conjugate the Hamiltonian if the operation is time-reversing, and apply the phase factor.
                expected = operation.conjugate ? conj(H[a, b]) * phase : H[a, b] * phase

                # Check the symmetry holds.
                @test H_lambda[mapped_a, mapped_b] ≈ expected atol = 1e-10
            end
        end
    end
end

@testset "Symmetry relation on H(k) with a fourfold axis" begin
    """
    The same relation as above, on a cell whose operations are not all their own inverse.

    Every cell tested elsewhere has an all-involution Laue group, mmm or 2/m, where R = R^{-1} and
    k_Λ = R k cannot be told apart from k_Λ = R^{-1} k. A fourfold axis separates them.
    """

    T = Float64

    # Tetragonal cell, four molecules related by successive quarter turns about z.
    lattice = SF.build_crystal_lattice(T[10 0 0; 0 10 0; 0 0 14], T)
    quarter_turn = Quaternionic.rotor(T[cos(π / 4), 0, 0, sin(π / 4)])
    turns = [quarter_turn^(index - 1) for index in 1:4]
    base = SVector{3, T}(3, 1, 0)
    images = [SF.CrystalImage{T}(1, turns[index], one(T),
                                 SVector{3, T}(Quaternionic.to_rotation_matrix(turns[index]) * base))
              for index in 1:4]
    operations = SF.derive_symmetry_operations(images, lattice)

    # The cell has to actually supply an operation that is not its own inverse.
    @test any(maximum(abs, operation.R_Lambda * operation.R_Lambda - I) > 1e-10
              for operation in operations)

    basis = BH.build_excitation_basis(images, [T[4.10, 4.65]])
    n_lambda_i = length(basis.Delta_E_monomer)

    # Build D from rotated densities, so the symmetry check includes its pair contractions as
    # well as its placement in H. Only molecules in this cell contribute to this finite reference.
    l_max = 2
    q_grid = collect(range(zero(T), T(6), length = 121))
    rotated_R, rotated_Delta, rotated_Xi = SF.rotate_crystal_R_tensors(
        [neutral_test_R(2, q_grid, l_max)], [neutral_test_R(3, q_grid, l_max)],
        [neutral_test_R(1, q_grid, l_max)], images, basis, l_max)
    cells = CJ.NeighbourCells{T}([SVector{3, T}(0, 0, 0)], [(0, 0, 0)], [1])
    mktempdir() do tmp_dir
        gaunt_path = joinpath(tmp_dir, "fourfold_D_gaunt.h5")
        SF.PrecomputeGaunt.precompute_gaunt_coefficients(l_max, 2 * l_max, l_max, gaunt_path, T)
        _, D_blocks = CJ.compute_crystal_corrections(
            rotated_R, rotated_Delta, rotated_Xi, basis, cells, q_grid, l_max, gaunt_path)
        @test abs(D_blocks[1, 1, 2]) > 1e-6
        for image_idx in 2:length(images)
            @test D_blocks[image_idx, :, :] ≈ D_blocks[1, :, :] rtol = 2e-12 atol = 2e-12
        end
        BH.set_D_couplings!(basis, D_blocks)
    end

    couplings = dipole_couplings(basis, images, lattice,
        [SVector{3, T}[SVector{3, T}(0.9, 0.2, -0.3), SVector{3, T}(-0.1, 0.7, 0.5)]])

    at_k = BH.BlochEigensystem(basis)
    at_k_lambda = BH.BlochEigensystem(basis)

    for k in (SVector{3, T}(0.11, -0.05, 0.13), SVector{3, T}(-0.07, 0.17, 0.09))
        BH.build_bloch_hamiltonian!(at_k, basis, k, couplings)
        H = Matrix(Hermitian(copy(at_k.H), :U))

        for operation in operations
            reversal = operation.conjugate ? -one(T) : one(T)
            k_lambda = reversal * (operation.R_Lambda * k)
            BH.build_bloch_hamiltonian!(at_k_lambda, basis, k_lambda, couplings)
            H_lambda = Matrix(Hermitian(copy(at_k_lambda.H), :U))

            for a in 1:n_lambda_i, b in 1:n_lambda_i
                mapped_a = 0
                mapped_b = 0
                for lambda in 1:n_lambda_i
                    basis.transition_of[lambda] == basis.transition_of[a] &&
                        basis.image_of[lambda] == operation.image_of[basis.image_of[a]] &&
                        (mapped_a = lambda)
                    basis.transition_of[lambda] == basis.transition_of[b] &&
                        basis.image_of[lambda] == operation.image_of[basis.image_of[b]] &&
                        (mapped_b = lambda)
                end

                phase = cis(-dot(k_lambda, operation.L_offsets[basis.image_of[a]] -
                                           operation.L_offsets[basis.image_of[b]]))
                expected = operation.conjugate ? conj(H[a, b]) * phase : H[a, b] * phase

                @test H_lambda[mapped_a, mapped_b] ≈ expected atol = 1e-10
            end
        end
    end
end


@testset "Gaunt convention" begin
    mktempdir() do directory
        path = joinpath(directory, "Gaunt.h5")
        SF.PrecomputeGaunt.precompute_gaunt_coefficients(1, 2, 1, path, Float64)
        for contraction in (SF.ConstructRTensor, SF.ConstructRTensorGPU, SF.ConstructFSquaredTensor)
            gaunt = contraction.load_gaunt_array(path, Float64)
            for m in -1:1
                # Y_0^0 = 1/√(4π), so coupling it to Y_1^m returns the same Y_1^m.
                index = findfirst(i -> gaunt.l1[i] == 1 && gaunt.l2[i] == 0 &&
                    gaunt.l3[i] == 1 && gaunt.m1[i] == m && gaunt.m3[i] == m,
                    eachindex(gaunt.Gaunt_values))
                @test index !== nothing
                @test gaunt.Gaunt_values[index] ≈ 1 / sqrt(4π)

                # ∫dΩ |Y_1^m|² Y_2^0 gives the Gaunt coefficient with plane-wave M = 0.
                index = findfirst(i -> gaunt.l1[i] == 1 && gaunt.l2[i] == 2 &&
                    gaunt.l3[i] == 1 && gaunt.m1[i] == m && gaunt.m3[i] == m,
                    eachindex(gaunt.Gaunt_values))
                @test index !== nothing
                expected = (m == 0 ? 2.0 : -1.0) * sqrt(5 / (4π)) / 5
                @test gaunt.Gaunt_values[index] ≈ expected
            end
        end

        # P̄ uses the Gaunt directly, with only i^L and the polar harmonic left to multiply.
        P_pair = CJ.build_pair_frame_P(path, Float64)
        for mu in -1:1
            triple = findfirst(==((1, 1, mu)), P_pair.triples)
            members = P_pair.offsets[triple]:(P_pair.offsets[triple + 1] - 1)
            for member in members
                L = P_pair.member_L[member]
                if L == 0
                    @test P_pair.member_weight[member] ≈ 1 / (4π)
                elseif L == 2
                    expected = -(mu == 0 ? 2.0 : -1.0) / (4π)
                    @test P_pair.member_weight[member] ≈ expected
                else
                    @test false
                end
            end
        end
    end
end

@testset "Fourier Parseval check with transition views" begin
    fft = SCarFFF.FFTFormFactor.PerformFFT
    density = reshape(collect(1.0:125.0), 1, 5, 5, 5)
    limits = [1.0, 2.0, 3.0]
    counts = [5, 5, 5]
    amplitude = fft.perform_fft(density, limits, counts)
    @test sum(abs2, amplitude) / prod(counts) ≈ sum(abs2, density) * prod(2 .* limits ./ counts)^2
    mktemp() do _, output
        redirect_stdout(output) do
            @test fft.check_parseval_theorem(view(density, 1, :, :, :),
                view(amplitude, 1, :, :, :), limits, counts) === nothing
        end
    end
end

@testset "Independent short-range q grid" begin
    for T in (Float32, Float64)
        # A sufficient saved grid needs no new form factors.
        saved_q = collect(range(zero(T), T(4); length = 21))
        q, indices = SF.reuse_short_range_grid(saved_q, T(0.4), T(3))
        @test q === saved_q
        @test indices == collect(eachindex(saved_q))

        # Extending a commensurate grid keeps every coincident sample.
        saved_q = T[0, 0.2, 0.4, 0.6, 0.8]
        q, indices = SF.reuse_short_range_grid(saved_q, T(0.4), T(1.6))
        @test q ≈ T[0, 0.4, 0.8, 1.2, 1.6]
        @test indices == [1, 3, 5, 0, 0]
        @test q[end] >= T(1.6)
        @test q[2] - q[1] <= T(0.4)

        # A long grid with insufficient resolution still needs new nodes.
        coarse_q = T[0, 1, 2, 3]
        q, indices = SF.reuse_short_range_grid(coarse_q, T(0.4), T(2))
        @test any(iszero, indices)
        @test q[2] - q[1] <= T(0.4)
        @test q[end] >= T(2)
        for index in findall(!iszero, indices)
            @test isapprox(q[index], coarse_q[indices[index]]; atol = 8 * eps(T))
        end

        mktempdir() do tmp_dir
            small_case = build_small_R_case(T, tmp_dir)
            W, sigma_ij, R_ij_mod, R_ij_hat, _, l_max, gaunt_path, _, _ = small_case.inputs
            mol = small_case.mol
            densities = Matrix{T}[mol.T_gs[1], mol.T_gg]
            append!(densities, RB.build_Delta_T_matrices(mol, [1]))
            envelope = SF.short_range_envelope(W, sigma_ij, mol, densities, 1)
            data = (; mol, W_ij = W, sigma_ij, R_ij_mod, R_ij_hat, density_matrices = densities,
                    gaunt_path, use_gpu = false, threshold = zero(T), envelope)
            saved_q = T[0, 0.4, 0.8]
            all_R = RT.construct_R_tensor(W, sigma_ij, R_ij_mod, R_ij_hat,
                saved_q, l_max, gaunt_path, densities, mol.cartesian_term_to_orbital)
            saved = DF.split_density_R_tensors(all_R, 1, mol.atom_coordinates,
                mol.nuclear_charges, saved_q, l_max)
            q, indices = SF.reuse_short_range_grid(saved_q, T(0.4), T(1.6))
            refined = SF.short_range_coefficients(data, q, indices, saved..., l_max)

            # The merged densities agree with evaluating the entire grid afresh.
            full_R = RT.construct_R_tensor(W, sigma_ij, R_ij_mod, R_ij_hat,
                q, l_max, gaunt_path, densities, mol.cartesian_term_to_orbital)
            reference = DF.split_density_R_tensors(full_R, 1, mol.atom_coordinates,
                mol.nuclear_charges, q, l_max)
            for family in 1:3
                @test refined[family] ≈ reference[family] rtol = 20 * eps(T) atol = 20 * eps(T)
                @test refined[family][:, 1:3, :] == saved[family]
            end
            @test all(iszero, refined[2][:, 1, 1])
            @test iszero(refined[3][1, 1, 1])

            # Removing W proves that a complete cache can be used without evaluating any FFs.
            cached_data = merge(data, (W_ij = nothing,))
            cached = SF.short_range_coefficients(cached_data, saved_q, [1, 2, 3], saved..., l_max)
            @test all(cached[family] == saved[family] for family in 1:3)

            rotation = Quaternionic.rotor(T[1, 0, 0, 0])
            image = SF.CrystalImage{T}(1, rotation, one(T), SVector{3, T}(0, 0, 0))
            basis = SF.build_excitation_basis([image], [T[2]])
            lattice = SF.build_crystal_lattice(T[4 0 0; 0 4 0; 0 0 4], T)
            cells = CJ.enumerate_neighbour_cells(lattice, T(5))
            parameters = SF.Ewald.EwaldParameters{T}(T(0.45), T(5), T(3), T(1e-3))
            q_loose, _ = SF.choose_short_range_grid(
                saved_q, [data], basis, cells, parameters, parameters, T(0.1), T(0.1))
            parameters = SF.Ewald.EwaldParameters{T}(T(0.45), T(5), T(3), T(1e-6))
            q_tight, _ = SF.choose_short_range_grid(
                saved_q, [data], basis, cells, parameters, parameters, T(0.1), T(0.1))
            @test q_tight[end] >= q_loose[end]
            @test q_tight[2] - q_tight[1] <= q_loose[2] - q_loose[1]

            zero_envelope = merge(envelope, (A = zeros(T, size(envelope.A)),))
            zero_data = merge(data, (envelope = zero_envelope,))
            q_zero, _ = SF.choose_short_range_grid(
                saved_q, [zero_data], basis, cells, parameters, parameters, zero(T), zero(T))
            @test q_zero === saved_q
        end
    end
end
