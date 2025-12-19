module PerformFFTGPU

using CUDA
using CUDA.CUFFT
using FFTW
using AbstractFFTs

export perform_fft_gpu

function fftshift_kernel!(
        output::CuDeviceArray{Complex{T}, 3},
        input::CuDeviceArray{Complex{T}, 3},
        nx::Int32,
        ny::Int32,
        nz::Int32
    ) where {T<:AbstractFloat}
    """
    A CUDA kernel to perform the fftshift on a 3D array. This shifts the zero-frequency component to the center.

    # Arguments:
    - output::CuDeviceArray{Complex{T}, 3}: The output array to store the shifted data.
    - input::CuDeviceArray{Complex{T}, 3}: The input array to shift.
    - nx::Int32: The number of grid points in the x dimension.
    - ny::Int32: The number of grid points in the y dimension.
    - nz::Int32: The number of grid points in the z dimension.
    """

    # Get the global thread index and stride.
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    total = nx * ny * nz

    @inbounds while idx <= total
        # Decode the 1D index into (i, j, k) coordinates.
        idx0 = idx - 1
        k = idx0 ÷ (nx * ny) + 1
        rem_xy = idx0 - (k - 1) * (nx * ny)
        j = rem_xy ÷ nx + 1
        i = rem_xy - (j - 1) * nx + 1

        # Compute the shifted indices.
        i_shift = mod(i + nx ÷ 2 - 1, nx) + 1
        j_shift = mod(j + ny ÷ 2 - 1, ny) + 1
        k_shift = mod(k + nz ÷ 2 - 1, nz) + 1

        # Copy the value to the shifted position.
        output[i_shift, j_shift, k_shift] = input[i, j, k]

        idx += stride
    end
end

function perform_fft_gpu(
        transition_densities::CuArray{T, 4},
        r_lim::Vector{U},
        N_grid::Vector{Int}
    )::CuArray{Complex{T}, 4} where {T<:AbstractFloat, U<:AbstractFloat}
    """
    Perform a 3D FFT on the transition density data on the GPU to get the form factor.

    # Arguments:
    - transition_densities::CuArray{T, 4}: Transition density data on the GPU with shape (n_transitions, n_x, n_y, n_z).
    - r_lim::Vector{U}: The limits of the grid in real space, in Angstroms.
    - N_grid::Vector{Int}: The number of grid points in each dimension.

    # Returns:
    - CuArray{Complex{T}, 4}: The form factors on the GPU with shape (n_transitions, n_qx, n_qy, n_qz).
    """

    n_transitions = size(transition_densities, 1)

    r_res = 2 .* r_lim ./ N_grid

    xs = T.(collect(range(-r_lim[1], step = r_res[1], length=N_grid[1])))
    ys = T.(collect(range(-r_lim[2], step = r_res[2], length=N_grid[2])))
    zs = T.(collect(range(-r_lim[3], step = r_res[3], length=N_grid[3])))

    qxs = T(2π) .* T.(collect(fftfreq(N_grid[1], 1/r_res[1])))
    qys = T(2π) .* T.(collect(fftfreq(N_grid[2], 1/r_res[2])))
    qzs = T(2π) .* T.(collect(fftfreq(N_grid[3], 1/r_res[3])))

    dV = prod(r_res)

    qxs_gpu = CuArray(qxs)
    qys_gpu = CuArray(qys)
    qzs_gpu = CuArray(qzs)

    # Precompute the phase factor on the GPU.
    phase_gpu = exp.(im .* (reshape(qxs_gpu, N_grid[1], 1, 1) .* xs[1] .+
                            reshape(qys_gpu, 1, N_grid[2], 1) .* ys[1] .+
                            reshape(qzs_gpu, 1, 1, N_grid[3]) .* zs[1]))

    # Allocate the output on GPU.
    form_factors_gpu = CUDA.zeros(Complex{T}, n_transitions, N_grid[1], N_grid[2], N_grid[3])

    # Allocate temporary buffers on GPU.
    td_gpu = CuArray{Complex{T}}(undef, N_grid[1], N_grid[2], N_grid[3])
    td_shifted_gpu = CuArray{Complex{T}}(undef, N_grid[1], N_grid[2], N_grid[3])

    # Setup the kernel parameters for fftshift.
    threads = 256
    total = N_grid[1] * N_grid[2] * N_grid[3]
    blocks = cld(total, threads)

    for t in 1:n_transitions
        # Copy transition density to temporary buffer and convert to complex.
        copyto!(td_gpu, complex.(view(transition_densities, t, :, :, :)))
        CUDA.synchronize()

        # Perform FFT in place.
        CUFFT.fft!(td_gpu)
        CUDA.synchronize()

        # Conjugate to partially correct phase and apply phase factor and volume element.
        conj!(td_gpu)
        td_gpu .*= phase_gpu
        td_gpu .*= T(dV)
        CUDA.synchronize()

        # Perform fftshift on GPU.
        @cuda threads=threads blocks=blocks fftshift_kernel!(td_shifted_gpu, td_gpu, Int32(N_grid[1]), Int32(N_grid[2]), Int32(N_grid[3]))
        CUDA.synchronize()

        # Store result in output array.
        copyto!(view(form_factors_gpu, t, :, :, :), td_shifted_gpu)
        CUDA.synchronize()
    end

    return form_factors_gpu
end

end
