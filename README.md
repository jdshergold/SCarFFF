# SCarFFF
Computation of form factors for scattering on molecules and crystals.

## Quickstart

1. **Clone the repository:**

   ```bash
    git clone https://github.com/jdshergold/SCarFFF.git
    cd SCarFFF
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

4. **Modify and run a job:**

   Edit one of the examples in `jobs/` to set the run parameters, or create your own job YAML.

   Then run:

   ```bash
   python3 run.py jobs/[job_name].yaml
   ```

   This will perform the TD-DFT calculation and compute the form factors or structure functions as requested. These will be saved in the `runs/[run_name]/1,2,...` directory structure.

5. **Plot the results:**

   Plotting is a separate command, so figures can be redrawn without recomputing anything:

   ```bash
   python3 plot.py runs/[run_name]
   ```

   The figures are saved in each numbered system's `plots/` directory. Which figures are drawn, and at what size, is set in `config/plotting.yaml`, which should be edited according to preference. You can also create additional plotting config files, which can be called when running the plotting script. See [the plotting guide](docs/plotting.pdf)
   for more details on plotting, including a gallery of every figure.

## Calculation jobs

There are four small examples: [spherical](jobs/spherical.yaml), [Fourier](jobs/fourier.yaml),
[Cartesian](jobs/cartesian.yaml) and [crystal](jobs/crystal.yaml). Only include sections
that apply to the selected `mode`. Paths are relative to the job YAML, including `output`.
The resolved settings are copied to `job.yaml` in the run directory.

Molecular input is either `input: {smiles: C1=CC=CC=C1}` or
`input: {csv: ../examples/uniform_set.csv}`. The CSV needs a `smiles` column. Molecules
are numbered in CSV row order. Crystal input uses `cif` or `cif_directory`; CIF files
in a directory are numbered in sorted filename order. Crystal mode uses the spherical
calculation and retains the existing conformer and crystal output structure.

Set **both** stage controls explicitly:

```yaml
stages:
  tddft: reuse
  form_factors: run
```

- `run` recomputes the stage.
- `reuse` validates the saved inputs, settings and required results. A mismatch is an
  error; it never starts another calculation.
- `skip` omits the stage. Form factors require TD-DFT to be `run` or `reuse`.

`nstates` sets the number of saved excited states. TD-DFT computes `nstates + 3`,
including a three-state buffer, and saves only the first `nstates` energies,
oscillator strengths and X and Y amplitudes. The `nstates_td_dft` attribute records
the computed count. `transitions: [1, 3, 4]` optionally selects form factors from
those saved states; the default is `all`. The same count applies to each crystal
conformer. Failed entries are reported, while other entries continue.

The method section specifies momentum grids in **keV**. Spherical `n_theta` and
`n_phi` default to `max(3, 4*l_max + 1)`; zero requests this automatic choice.
Spherical `save` selects `f_s`, `f_sq_lm` and/or `R_tensor`. Cartesian `save` selects
`f_s` and/or shared `V` tensors. Fourier results contain complex `f_s` and transition
densities. Fourier `q_max` and `dq` each contain three axis values; the saved momentum
coordinates record the actual FFT frequencies, including asymmetric endpoints for
an even number of samples.

Other supported settings include:

```yaml
execution:
  precision: float64          # float32 or float64
  gpu: false
  threads: 4                 # or auto
verification:
  oscillator_strength: true  # Requires saved f_s
  q_max_fit: 0.05             # keV; include at least two low-q samples
```

Fourier jobs can also set `fourier.check_parseval: true`.

The resolved YAML records every default, including crystal coupling/Ewald parameters,
energy resolution and sparse tensor thresholds. Unknown parameters
are rejected. `spherical.threshold` and `cartesian.threshold` default to `1.0e-6`.
`execution.threads` sets Julia threads and, when numeric, the PySCF thread count.

## Saved molecular results

Each run has one directory per molecule, regardless of the method:

```text
runs/benzene/
  job.yaml
  metadata.csv
  1/
    td_dft.h5
    form_factors.h5
    plots/
  2/
    ...
```

The TD-DFT dataset layout is unchanged. The selected molecular method has one
form-factor file:

```text
form_factors.h5
  grid/                       # Shared q/angular coordinates, or q_x/q_y/q_z
  transitions/1/
    Delta_E                   # eV
    f_s                       # Complex amplitude, when requested
    f_sq_lm                   # Real intensity coefficients, when requested and available
    R_tensor                  # When requested
    transition_density        # Fourier only, Angstrom^-3
  transitions/2/
    ...
  cartesian_tensors/          # Optional shared V_x, V_y and V_z
```

Fourier spatial coordinates are also shared under `grid/`. Dimensional units are
dataset attributes. Dimensionless quantities have no units attribute. Dtypes retain
the requested precision; Julia/Python array ordering is handled in the plotting
reader. Molecular files are written through a temporary file and replaced only
after writing and any requested verification finish successfully.

Existing saved runs remain untouched. The new molecular interface reads the grouped
format only. The previous plotters have been retired: of the plots they drew, the
transition density is now part of the new interface, and the band maps, rates, crystal
form-factor modes and 3D views have been removed.

## Supported Basis Sets

All PySCF orbital basis sets with angular momentum up to l = 6 (i-orbitals) are supported.
Set the basis set name with `tddft.basis` in the job YAML.

### Naming convention

The basis string is matched to a precomputed HDF5 file by a canonical transformation: lowercase, replace `*` with `s`, strip `-`, `_`, spaces, and parentheses. Common variant spellings all resolve to the same file:

| You write | Resolves to |
| --- | --- |
| `cc-pVDZ` | `ccpvdz.h5` |
| `6-31G*` | `631gs.h5` |
| `def2-TZVP` | `def2tzvp.h5` |
| `aug-cc-pVTZ` | `augccpvtz.h5` |

Pople polarisation and diffuse variants follow PySCF's extended naming scheme — e.g. `6-31g(2d,2p)` canonicalises to `631gpolarization2d`. Use the canonical form or pass the PySCF filename stem directly. There are lots of basis sets, and not every one has been tested. Please let us know if you find an issue!

## File Structure

```text
SCarFFF/
├── run.py                     # Runs a job: TD-DFT, then the form factors.
├── plot.py                    # Draws the figures from a finished run.
│
├── jobs/                      # Example job YAMLs, one per method.
│   ├── spherical.yaml         # Spherical harmonic expansion, for a molecule.
│   ├── cartesian.yaml         # Cartesian grid.
│   ├── fourier.yaml           # FFT of the real space transition density.
│   └── crystal.yaml           # Crystal, which uses the spherical method per molecule.
│
├── config/
│   └── plotting.yaml          # Which figures to draw, at what size and in what format.
│
├── docs/                      # The plotting guide.
│
├── src/
│   ├── SCarFFF.jl             # Main Julia package module, with precompilation and exports.
│   ├── results.py             # HDF5 metadata reading shared by the Python and Julia sides.
│   │
│   ├── jobs/                  # Everything that turns a job YAML into saved results.
│   │   ├── config.py          # Reads and validates the job YAML, filling in the defaults.
│   │   ├── runner.py          # Runs the stages, and validates a reused one against its settings.
│   │   ├── tddft.py           # TD-DFT with PySCF and RDKit, for molecules and crystal conformers.
│   │   ├── crystal.py         # Reads a CIF, and groups its molecules into conformers.
│   │   ├── compute.jl         # The molecular form factor stage.
│   │   ├── CrystalJob.jl      # The crystal stage: conformers, couplings, bands and the structure function.
│   │   ├── Results.jl         # Writes a molecular result file, through a temporary file.
│   │   └── Verification.jl    # Checks the form factor's dipole against the PySCF oscillator strength.
│   │
│   ├── plotting/              # Everything that turns saved results into figures.
│   │   ├── main.py            # Picks the saved systems and the figures that apply to them.
│   │   ├── config.py          # Reads and validates the plotting YAML.
│   │   ├── data.py            # Reads saved quantities, and takes angular moments and planes.
│   │   ├── figures.py         # The individual molecular and structure function plots.
│   │   ├── overviews.py       # The four panel molecular and crystal overviews.
│   │   └── style.py           # Computer Modern styling and the shared print layouts.
│   │
│   ├── precomputation/        # Precomputing angular momentum coefficients.
│   │   ├── PrecomputeGaunt.jl # Gaunt coefficient computation and HDF5 storage.
│   │   └── PrecomputeATensor.jl # A tensor computation and HDF5 storage.
│   │
│   ├── utils/                 # Utility modules for I/O, basis sets, and helper functions.
│   │   ├── ReadBasisSet.jl    # Reads GTO basis sets and constructs molecular data from HDF5.
│   │   ├── DensityMatrices.jl # AO density matrices from the PySCF X and Y amplitudes.
│   │   ├── SparseTensors.jl   # Sparse tensor data structures (COO format).
│   │   ├── BinEncoding.jl     # HDF5 bin encoding/decoding for vectors of vectors.
│   │   ├── FastPowers.jl      # Fast integer power functions.
│   │   ├── ThreadChunks.jl    # Splits a loop range into contiguous per-task chunks.
│   │   └── StageTimings.jl    # Times each stage and prints the breakdown.
│   │
│   ├── form_factors/          # Form factor calculation code.
│   │   ├── common/            # Shared form factor libraries.
│   │   │   ├── ConstructPairCoefficients.jl # Pair coefficients for Cartesian term combinations.
│   │   │   └── ConstructbCoefficients.jl    # b coefficients for spherical expansion.
│   │   │
│   │   ├── spherical/         # Spherical harmonic expansion method, and the crystal treatment.
│   │   │   ├── SphericalFormFactor.jl       # Main module for the spherical method.
│   │   │   ├── ConstructDTensor.jl          # D tensor construction.
│   │   │   ├── ConstructWTensor.jl          # W tensor construction.
│   │   │   ├── ConstructRTensor.jl          # R tensor construction (CPU).
│   │   │   ├── ConstructRTensorGPU.jl       # R tensor construction (GPU).
│   │   │   ├── ConstructFSquaredTensor.jl   # fsq tensor construction (CPU).
│   │   │   ├── ContractSphericalGrid.jl     # Final contraction to spherical grid (CPU).
│   │   │   ├── ContractSphericalGridGPU.jl  # Final contraction to spherical grid (GPU).
│   │   │   ├── DensityFormFactors.jl        # The extra form factors the same-molecule couplings D need.
│   │   │   ├── ProjectFSquared.jl           # Projects |f|^2 on a (q, θ, ϕ) grid onto real spherical harmonics.
│   │   │   ├── WignerRotations.jl           # Wigner D blocks, which rotate spherical harmonic coefficients.
│   │   │   ├── CrystalLattice.jl            # Lattice geometry, and folding q into the first Brillouin zone.
│   │   │   ├── CouplingJ.jl                 # The coupling, J, between excitations on different molecules.
│   │   │   ├── Ewald.jl                     # The Ewald split of the intermolecular Coulomb coupling.
│   │   │   ├── BlochHamiltonian.jl          # Builds and diagonalises the Frenkel exciton Bloch Hamiltonian.
│   │   │   ├── CrystalSymmetry.jl           # Finds the symmetry operations that cover a whole star of k points.
│   │   │   ├── CoherentCrystalFormFactor.jl # Mixes the rotated monomer form factors with the Bloch coefficients.
│   │   │   └── StructureFactor.jl           # The energy axis, and the broadened delta that puts each state on it.
│   │   │
│   │   ├── cartesian/         # Cartesian grid method using Hermite polynomials.
│   │   │   ├── CartesianFormFactor.jl       # Main module for the Cartesian method.
│   │   │   ├── ProbabilistsHermite.jl       # Fast implementation of Hermite polynomials.
│   │   │   ├── ConstructVTensors.jl         # V tensor construction.
│   │   │   ├── ContractCartesianGrid.jl     # Final contraction to Cartesian grid (CPU).
│   │   │   └── ContractCartesianGridGPU.jl  # Final contraction to Cartesian grid (GPU).
│   │   │
│   │   └── fft/               # FFT method.
│   │       ├── FFTFormFactor.jl             # Main module for the FFT method.
│   │       ├── ConstructIntegrand.jl        # Transition density construction in real space (CPU).
│   │       ├── ConstructIntegrandGPU.jl     # Transition density construction in real space (GPU).
│   │       ├── PerformFFT.jl                # Performs the FFT of the transition density (CPU).
│   │       └── PerformFFTGPU.jl             # Performs the FFT of the transition density (GPU).
│   │
│   └── data/                  # Precomputed data. Basis set files, Gaunt coefficients, A tensors.
│       ├── basis_sets/        # Gaussian basis set definitions (HDF5).
│       │   └── {basis_set_name}.h5
│       ├── gaunt_coefficients/ # Precomputed Gaunt coefficients (HDF5).
│       │   └── gaunt_coefficients_*.h5
│       └── A_tensors/         # Precomputed A tensors (HDF5).
│           └── A_tensor_*.h5
│
├── test/
│   ├── runtests.jl            # The Julia numerical tests.
│   ├── test_job_io.py         # The Python job, IO and plotting-data tests.
│   └── write_test_results.jl  # Writes their HDF5 fixtures with the real Julia writer.
│
└── runs/                      # Output directory for computations.
    └── [run_name]/            # Organised by run name.
        ├── job.yaml           # The resolved settings, with every default recorded.
        ├── metadata.csv       # One row per molecule or crystal.
        └── 1, 2, .../         # One directory per molecule or crystal, numbered by input order.
            ├── td_dft.h5             # Molecular only: energies, oscillator strengths, X and Y.
            ├── form_factors.h5       # Molecular only: the grids and one group per transition.
            ├── crystal_metadata.json # Crystal only: the lattice, conformers and image rotations.
            ├── conformers/[label]/   # Crystal only: per conformer TD-DFT and R tensors.
            ├── crystal/              # Crystal only: crystal_form_factors_[precision].h5.
            └── plots/                # The figures, in subdirectories by page width.
```
