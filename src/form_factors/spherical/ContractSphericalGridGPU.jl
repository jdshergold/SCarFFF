# This module contains GPU functions to contract the angular and radial parts of the form factor.

module ContractSphericalGridGPU

export contract_spherical_grid_gpu

using CUDA
using ..ConstructRTensorGPU: precompute_spherical_harmonic_normalisation
using ...FastPowers: fast_neg1_pow

function contract_spherical_grid_gpu(
    R_pos::CuArray{Complex{T}},
    R_neg::CuArray{Complex{T}},
    theta_grid::Vector{T},
    phi_grid::Vector{T},
) where {T<:AbstractFloat}
    """
    Contract the radial and angular parts of the form factor to obtain f_s(q, θ, ϕ) on a spherical grid, defined by:

        f_s(q, θ, ϕ) = ∑_{ℓ=0}^{l_max} ∑_{m=-ℓ}^{ℓ} R_{ℓm}(q) Y_{ℓm}(θ, ϕ),

    where R is the R tensor that depends only on the modulus of the momentum transfer, q,
    and Y_{ℓm} are the spherical harmonics. The √2 is from spin-degeneracy.

    This GPU implementation computes Y_{ℓm}(θ, ϕ) for m ≥ 0 only, then performs two dense matrix
    multiplications (GEMMs) to exploit the symmetry Y_{ℓ,-m}(θ, ϕ) = (-1)^m conj(Y_{ℓm}(θ, ϕ)).

    # Arguments:
    - R_pos::CuArray{Complex{T}}: The R tensor for m ≥ 0 with dimensions (n_transitions, n_q, n_keys_pos).
    - R_neg::CuArray{Complex{T}}: The R tensor for m < 0 with dimensions (n_transitions, n_q, n_keys_pos), keyed by |m|.
    - theta_grid::Vector{T}: The θ values at which to evaluate the form factor, in radians.
    - phi_grid::Vector{T}: The ϕ values at which to evaluate the form factor, in radians.

    # Returns:
    - f_s::CuArray{Complex{T}, 4}: The spherical form factor evaluated on the grid with dimensions (n_transitions, n_q, n_θ, n_ϕ).
    """

    # Grid the sizes and dimensions.
    n_transitions = size(R_pos, 1)
    n_theta = length(theta_grid)
    n_phi = length(phi_grid)
    n_q = size(R_pos, 2)
    n_bins = n_theta * n_phi

    # Determine l_max from the key dimension (m ≥ 0 only).
    n_keys_pos = size(R_pos, 3)
    l_max = Int(round(-1.5 + sqrt(2 * n_keys_pos + 0.25)))

    # Move the angular grids to the GPU.
    theta_grid_gpu = CuArray(theta_grid)
    phi_grid_gpu = CuArray(phi_grid)

    # Precompute the Ylm normalisation and (-1)^m factors on the CPU.
    Ylm_norms = precompute_spherical_harmonic_normalisation(T, l_max)
    Ylm_norms_gpu = CuArray(Ylm_norms)
    neg1_key_cpu = Vector{T}(undef, n_keys_pos)
    @inbounds for l in 0:l_max
        key_base = (l * (l + 1)) ÷ 2 + 1
        for m in 0:l
            key = key_base + m
            neg1_key_cpu[key] = fast_neg1_pow(m, T)
        end
    end
    neg1_key_gpu = CuArray(neg1_key_cpu)

    # Precompute the exp(i m ϕ) phases for m = 0:l_max on the CPU and move to the GPU.
    phase_table_cpu = Array{Complex{T}, 2}(undef, l_max + 1, n_phi)
    @inbounds for phi_idx in 1:n_phi
        phi_val = phi_grid[phi_idx]
        for m in 0:l_max
            phase_table_cpu[m + 1, phi_idx] = Complex{T}(cos(phi_val * m), sin(phi_val * m))
        end
    end
    phase_table_gpu = CuArray(phase_table_cpu)

    # Warm up the CUDA kernels to avoid first-launch overhead in profiling.
    Ylm_norms_warm = CuArray{T}([one(T)])
    phase_table_warm = CUDA.ones(Complex{T}, 1, 1)
    theta_warm = CuArray{T}([zero(T)])
    @cuda threads=1 blocks=1 spherical_harmonics_grid_kernel!(CUDA.zeros(Complex{T}, 1, 1), Ylm_norms_warm, phase_table_warm, theta_warm, Int32(1), Int32(0), Int32(1))

    # Compute Y_L^M(θ, ϕ) directly on the GPU. Layout: (key_pos, bin).
    Ylm_matrix_gpu = CUDA.zeros(Complex{T}, n_keys_pos, n_bins)
    threads = 256
    total_bins = Int32(n_bins)
    blocks = cld(total_bins, threads)
    @cuda threads=threads blocks=blocks spherical_harmonics_grid_kernel!(Ylm_matrix_gpu, Ylm_norms_gpu, phase_table_gpu, theta_grid_gpu, Int32(n_theta), Int32(l_max), total_bins)

    # Contract R and Ylm to obtain f_s on the GPU using two GEMMs.
    # First we flatten the R tensors to 2D.
    R_pos_flat = reshape(R_pos, n_transitions * n_q, n_keys_pos)
    R_neg_flat = reshape(R_neg, n_transitions * n_q, n_keys_pos)

    # Next, we perform the first GEMM. This is R_lm * Y_lm for m ≥ 0.
    f_pos = R_pos_flat * Ylm_matrix_gpu

    # Next, we perform the second GEMM. This is (-1)^m R_lm * conj(Y_lm) for m < 0.
    R_neg_scaled = R_neg_flat .* reshape(neg1_key_gpu, (1, n_keys_pos))
    f_neg = R_neg_scaled * conj.(Ylm_matrix_gpu)

    # Finally, we sum the two contributions.
    f_gpu_flat = f_pos .+ f_neg

    # Reshape back to (transition, q, θ, ϕ).
    f_gpu = reshape(f_gpu_flat, n_transitions, n_q, n_theta, n_phi)
    return f_gpu
end

function spherical_harmonics_grid_kernel!(
    Ylm_tensor::CuDeviceMatrix{Complex{T}},
    Ylm_norms::CuDeviceVector{T},
    phase_table::CuDeviceMatrix{Complex{T}},
    theta_grid::CuDeviceVector{T},
    n_theta::Int32,
    L_max::Int32,
    total_angles::Int32,
) where {T<:AbstractFloat}
    """
    A kernel to compute Y_L^M(θ, ϕ) for all (θ, ϕ) bins on the GPU, storing only M ≥ 0 in (key, bin) layout.
    Each thread computes all spherical harmonics for one (θ, ϕ) bin. The (L, M) spherical harmonics are
    stored with the linear key:

        key(L, M) = (L * (L + 1)) / 2 + M + 1.

    The associated Legendre polynomials are computed using the recurrence relations:

        P_0^0(x) = 1,
        P_M^M(cos θ) = -(2M - 1) sin(θ) P_{M-1}^{M-1}(cos θ),
        P_{M+1}^M(cos θ) = (2M + 1) cos(θ) P_M^M(cos θ),
        P_L^M(cos θ) = ((2L - 1) cos(θ) P_{L-1}^M(cos θ) - (L + M - 1) P_{L-2}^M(cos θ)) / (L - M).

    # Arguments:
    - Ylm_tensor::CuDeviceMatrix{Complex{T}}: The output tensor to store the spherical harmonics.
    - Ylm_norms::CuDeviceVector{T}: The precomputed normalisation constants for Y_L^M.
    - phase_table::CuDeviceMatrix{Complex{T}}: Precomputed exp(i M ϕ) phases for M = 0:L_max.
    - theta_grid::CuDeviceVector{T}: The θ values for the angular grid.
    - n_theta::Int32: The number of θ grid points.
    - L_max::Int32: The maximum L value to compute.
    - total_angles::Int32: The total number of (θ, ϕ) angles.
    """

    # Get the global thread index and stride.
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    @inbounds while idx <= total_angles
        bin_idx = idx
        # Decode the bin index into (θ, ϕ) indices.
        theta_idx = Int32(mod(bin_idx - 1, n_theta)) + 1
        phi_idx = Int32((bin_idx - 1) ÷ n_theta) + 1

        theta = theta_grid[theta_idx]
        st, ct = sincos(theta) # For our purposes, sqrt(1-x^2) = sin(θ).

        # Seed the recurrence for associated Legendre polynomials. P_0^0 = 1.
        P_MM_m1 = one(T)

        # Compute spherical harmonics for all (L, M) with M ≥ 0.
        @inbounds for M in Int32(0):L_max
            # Compute the diagonal term P_M^M.
            key = (M * (M + 1)) ÷ 2 + M + Int32(1)

            if M == 0
                P_MM = P_MM_m1
            else
                # P_M^M(cos θ) = -(2M - 1) sin(θ) P_{M-1}^{M-1}(cos θ).
                P_MM = -(2*M - 1) * st * P_MM_m1
                P_MM_m1 = P_MM
            end

            # Convert to the correct spherical harmonic and store Y_M^M.
            phase = phase_table[M + 1, phi_idx]
            Ylm_tensor[key, bin_idx] = Complex{T}(Ylm_norms[key] * P_MM, zero(T)) * phase

            # There are no terms with L > L_max.
            if M == L_max
                continue
            end

            # Compute the off-diagonal terms.
            P_LM_m2 = P_MM
            P_LM_m1 = (2*M + 1) * ct * P_MM # P_{M+1}^M(cos θ) = (2M + 1) cos(θ) P_M^M(cos θ).

            # Store Y_{M+1}^M.
            key_off = ((M + 1) * (M + 2)) ÷ 2 + M + Int32(1)
            Ylm_tensor[key_off, bin_idx] = Complex{T}(Ylm_norms[key_off] * P_LM_m1, zero(T)) * phase

            # Compute the remaining off-diagonal terms using the three-term recurrence.
            for L in (M + 2):L_max
                P_LM = ((2*L - 1) * ct * P_LM_m1 - (L + M - 1) * P_LM_m2) / (L - M)
                key_LM = (L * (L + 1)) ÷ 2 + M + Int32(1)
                Ylm_tensor[key_LM, bin_idx] = Complex{T}(Ylm_norms[key_LM] * P_LM, zero(T)) * phase

                # Update temporary variables for the next iteration.
                P_LM_m2 = P_LM_m1
                P_LM_m1 = P_LM
            end
        end

        idx += stride
    end
end

end
