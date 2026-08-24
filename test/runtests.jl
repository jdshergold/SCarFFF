using Test
using SCarFFF

const SF = SCarFFF.SphericalFormFactor
const RT = SF.ConstructRTensor

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
    return inputs
end

@testset "Shared Cartesian-pair R contraction" begin
    mktempdir() do tmp_dir
        for T in (Float32, Float64)
            inputs = build_small_R_case(T, tmp_dir)
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
