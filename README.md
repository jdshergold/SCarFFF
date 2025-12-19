# SCarFFF
Computation of form factors for scattering on molecules.

## Quickstart

1. **Clone the repository:**

   ```bash
    git clone https://github.com/jdshergold/dark_molecules.git
    cd dark_molecules
    ```

2. **Setup python environment and install dependencies:**

   ```bash
   python3 -m venv env
   source env/bin/activate
   python3 -m pip install -r requirements.txt
   ```

3. **Install Julia and required packages:**
    Follow the instructions at https://julialang.org/downloads/ to install Julia.

    Then, from the project root directory, run:

    ```bash
    julia --project=. -e 'using Pkg; Pkg.instantiate()'
    ```
    
4. **Modify and run the computation script:**

   Edit `scripts/run_computation.sh` to set the run parameters.

   Then run:

   ```bash
   bash scripts/run_computation.sh
   ```

   This will perform the TD-DFT calculation, compute the molecular form factors, and generate plots. These will be saved in the `runs/[run_name]/[molecule_number]/` directory structure.

## File Structure

```text
SCarFFF/
├── scripts/                   # The main executable scripts live here.
│   ├── td_dft.py              # Script to perform TD-DFT calculations using PySCF and RDKit.
│   ├── compute_form_factor.jl # Script to compute molecular form factors using the Julia library.
│   ├── plot_3d.py             # 3D plotting script for form factor data.
│   ├── plot_slices.py         # Script for plotting 2D slices of form factor data.
│   └── run_computation.sh     # Bash script for running the full computation pipeline.
│
├── src/                       # Core Julia library code.
│   ├── SCarFFF.jl             # Main package module with precompilation and exports.
│   │
│   ├── precomputation/        # Precomputing angular momentum coefficients.
│   │   ├── PrecomputeGaunt.jl # Gaunt coefficient computation and HDF5 storage.
│   │   └── PrecomputeATensor.jl # A tensor computation and HDF5 storage.
│   │
│   ├── utils/                 # Utility modules for I/O, basis sets, and helper functions.
│   │   ├── ReadBasisSet.jl    # Reads GTO basis sets and constructs molecular data from HDF5.
│   │   ├── SparseTensors.jl   # Sparse tensor data structures (COO format).
│   │   ├── BinEncoding.jl     # HDF5 bin encoding/decoding for vectors of vectors.
│   │   └── FastPowers.jl      # Fast integer power functions.
│   │
│   ├── form_factors/          # Form factor calculation code.
│   │   ├── common/            # Shared form factor libraries.
│   │   │   ├── ConstructPairCoefficients.jl # Pair coefficients for Cartesian term combinations.
│   │   │   ├── ConstructbCoefficients.jl    # b coefficients for spherical expansion.
│   │   │   └── CheckOscillatorStrength.jl   # Oscillator strength validation.
│   │   │
│   │   ├── spherical/         # Spherical harmonic expansion method.
│   │   │   ├── SphericalFormFactor.jl       # Main module for the spherical method.
│   │   │   ├── ConstructDTensor.jl          # D tensor construction.
│   │   │   ├── ConstructWTensor.jl          # W tensor construction.
│   │   │   ├── ConstructRTensor.jl          # R tensor construction (CPU).
│   │   │   ├── ConstructRTensorGPU.jl       # R tensor construction (GPU).
│   │   │   ├── ContractSphericalGrid.jl     # Final contraction to spherical grid (CPU).
│   │   │   └── ContractSphericalGridGPU.jl  # Final contraction to spherical grid (GPU).
│   │   │
│   │   ├── cartesian/         # Cartesian grid method using Hermite polynomials.
│   │   │   ├── CartesianFormFactor.jl       # Main module forthe Cartesian method.
│   │   │   ├── ProbabilistsHermite.jl       # Fast implementation of Hermite polynomials.
│   │   │   ├── ConstructVTensors.jl         # V tensor construction.
│   │   │   └── ContractCartesianGrid.jl     # Final contraction to Cartesian grid.
│   │   │
│   │   └── fft/               # FFT method.
│   │       ├── FFTFormFactor.jl             # Main module for the FFT method.
│   │       ├── ConstructIntegrand.jl        # Transition density construction in real space.
│   │       └── PerformFFT.jl                # Performs the FFT of the transition density.
│   │
│   └── data/                  # Precomputed data. Basis set files, Gaunt coefficients, A tensors.
│       ├── basis_sets/        # Gaussian basis set definitions.
│       │   ├── 6-31g_st.gbs
│       │   └── cc-pVDZ.gbs
│       ├── gaunt_coefficients/ # Precomputed Gaunt coefficients (HDF5).
│       │   └── gaunt_coefficients_*.h5
│       └── A_tensors/         # Precomputed A tensors (HDF5).
│           └── A_tensor_*.h5
│
└── runs/                      # Output directory for computations.
    └── [run_name]/            # Organised by run name.
        └── [molecule_number]/ # Individual molecule results.
            ├── td_dft_results_f32.h5  # TD-DFT results (HDF5 format).
            ├── molecule_3d.html       # 3D interactive molecule visualisation.
            │
            ├── spherical/     # Spherical method results.
            │   └── transition_[n]/
            │       ├── fs_grid_f32.h5 # Form factor on spherical grid.
            │       ├── fs_3d.html     # Interactive 3D form factor visualisation.
            │       └── *.png          # 2D slice plots.
            │
            ├── cartesian/     # Cartesian method results.
            │   └── transition_[n]/
            │       ├── fs_grid_f32.h5 # Form factor on Cartesian grid.
            │       ├── fs_3d.html     # Interactive 3D form factor visualisation.
            │       └── *.png          # 2D slice plots.
            │
            └── fft/           # FFT method results.
                └── transition_[n]/
                    ├── fs_grid_f32.h5 # Form factor from FFT.
                    ├── fs_3d.html     # Interactive 3D form factor visualisation.
                    └── *.png          # 2D slice plots.
```
