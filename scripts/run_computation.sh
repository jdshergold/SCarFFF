#!/bin/bash
# Pipeline script to run TD-DFT, compute form factors, and generate plots.

set -e  # Exit if anything breaks.

# ========================================
# CONFIG STARTS HERE
# ========================================

# Run configuration.
RUN_NAME="tstilb09"     # Name for this run.

# Input specification. Specify exactly one of the following four options.
# CIF_FILE and CIF_DIR are for crystal mode only, and require METHOD="spherical".
CSV_FILE=""                 # Batch molecule mode. Path to a CSV file of SMILES strings.
SMILES=""        # Single molecule mode. SMILES string for the molecule.
CIF_FILE="examples/our_favourite_molecules/TSTILB09.cif"  # Single crystal mode. Path to a CIF file.
CIF_DIR=""                  # Batch crystal mode. Path to a directory of CIF files.


# ==== TD-DFT parameters. ====
BASIS="sto-3g"             # Basis set to use. See README for supported basis sets and aliases.
XC_FUNCTIONAL="lda"        # Exchange-correlation functional for PySCF (e.g. b3lyp, wB97X-V).
NSTATES=12                  # Number of excited states to compute.
NTRANS=12                   # Number of transitions to analyse.
RING_FLATTEN="--no-ring-flatten"            # Set to "--no-ring-flatten" to flatten based on whole molecule, or leave as "" for ring-based flattening.
DFT_OPTIMISATION=false    # Set to true to perform a DFT geometry optimisation after RDKit. Can be very slow on the CPU for large molecules.
PLOT_MOLECULE_3D=false    # Set to true to generate interactive 3D molecule plot from TD-DFT.

# ==== Form factor computation parameters. ====
METHOD="spherical"         # Method for form factor computation. Options: "spherical", "fft", or "cartesian".

# ==== Spherical method parameters. ====
# Momentum transfer grid parameters.
Q_MAX=20.0                 # Maximum momentum transfer in keV.
N_Q=101                    # Number of |q| grid points.
N_THETA=101                # Number of theta (polar angle) grid points.
N_PHI=101                  # Number of phi (azimuthal angle) grid points.

# Spherical harmonic expansion parameters.
L_MAX=24                   # Maximum angular mode, l, to include in spherical harmonic expansion.
COMPUTE_MODES=("form_factor" "f_lm_tensor")  # What to compute/save for the spherical method. Options: form_factor, R_tensor, f_lm_tensor.

# ==== Rate computation parameters (spherical method only). ====
COMPUTE_RATES=true        # Set to true to compute DM scattering rates after the spherical form factor.
M_GRID="1.0,1000.0,50"   # DM mass grid to use, in the form min_MeV,max_MeV,N (log-spaced). Only used if COMPUTE_RATES=true.
N_ROTATIONS="12,6,12"     # Number of detector rotations to consider (n_alpha,n_beta,n_gamma). Only used if COMPUTE_RATES=true.

# ==== FFT method parameters. ====
Q_LIM="15.0,15.0,15.0"           # q-space limits in keV, comma-separated (qx_max, qy_max, qz_max).
Q_RES="0.075,0.075,0.075"        # q-space resolution in keV, comma-separated (Δqx, Δqy, Δqz).
CHECK_PARSEVAL=false       # Set to true to check Parseval's theorem holds.

# ==== Cartesian method parameters. ====
QX_GRID="-15,15,101"       # qx grid specification (min,max,N) in keV.
QY_GRID="-15,15,101"       # qy grid specification (min,max,N) in keV.
QZ_GRID="-15,15,101"       # qz grid specification (min,max,N) in keV.
CARTESIAN_COMPUTE_MODE="form_factor"  # What to compute/save for the Cartesian method. Options: "V_only", "form_factor", "both". If "V_only" is selected, the computation stops before the grid contraction, and only V tensors are saved to disk.

# ==== Computation parameters. ====
TRANSITION_INDICES="all"       # Which electronic transitions to compute. Can be "all" or a comma-separated list like "1,2,3,4". The first excited state is index 1.
THRESHOLD=1e-6             # Threshold for dropping small tensor values. For spherical: |W/W_max| < THRESHOLD. For Cartesian: |M_ij/M_max| < THRESHOLD. Set to 0.0 to disable. Around 1e-6 is recommended for both accuracy and performance.
JULIA_THREADS="auto"       # Number of threads to use for the Julia part of the code. Set to "auto" to use all available threads.
PRECISION="float32"       # Floating point precision for form factor computation. On CPU this has negligible impact on performance, but cuts down on file size by ~2. May produce 'fuzzy looking' Re/Im plots when the values are effectively zero in float32.
USE_GPU=false              # Set to true to enable GPU acceleration for the DFT and form factor computations. Supports spherical, FFT, and Cartesian methods.

# ==== Control flags. ====
SKIP_TDDFT=true            # Set to true to skip the TD-DFT calculation. Will not be skipped if the results do not already exist.
SKIP_FORM_FACTOR=false     # Set to true to skip the form factor computation. Will not be skipped if the results do not already exist.
SKIP_2D_PLOTS=false        # Set to true to skip 2D slice plot generation.
SKIP_3D_PLOTS=true         # Set to true to skip 3D isosurface plot generation.
FORCE_RECOMPUTATION=false  # Set to true to force recomputation of A and Gaunt coefficients in the form factor computation.
BENCHMARK=false            # Set to true to run benchmark after form factor computation. Testing only.

# ==== Verification. ====
CHECK_OSCILLATOR_STRENGTH=false  # Set to true to compute and compare oscillator strengths found from the form factor with those from PySCF.
Q_MAX_FIT=0.05                  # Maximum q value in keV for fitting the oscillator strength. You should ensure that at least 2 points sit below this value. It is highly recommended to set this to 0.2 keV or lower, the lower the better.

# ==== Plotting parameters. ====
PLOT_TRANSITION_DENSITY=true  # Set to true to plot the transition density (FFT method only, will be ignored for the spherical method).

# 2D plotting parameters.
PLOT_PLANES=("xy" "xz" "yz")  # Planes to plot. Options: xy (qz=0), xz (qy=0), yz (qx=0). Each option will be plotted as a column, in the same order as provided.
PLOT_MODES=("modsq" "Re" "Im")  # What to plot. Options: modsq (|f_s|^2), Re (real part), Im (imaginary part). Each option will be plotted as row, in the same order as provided.
PLOT_FLM_MODES=true           # Set to true to plot dominant f^2_{lm}(q) modes (spherical method only, requires f_lm_tensor in COMPUTE_MODES).

# 3D plotting parameters.
PLOT_3D_MODES=("modsq" "Re" "Im")  # Modes for 3D plots. Options: modsq (|f_s|^2), Re (real part), Im (imaginary part). Each mode will generate a separate 3D plot.
PLOT_3D_MIN_FRACTION=0.15 # Minimum isosurface level as a fraction of maximum.
PLOT_3D_MAX_FRACTION=0.95 # Maximum isosurface level as a fraction of maximum.
PLOT_3D_GRID_SIZE=75      # Target grid (x,y,z) size for downsampling. Setting this too large will make the plots laggy.

# Plot range parameters, for zooming in on specific regions. Leave empty to use the full range.
# The q ranges are in keV, the spatial ranges are in Angstroms.
PLOT_RANGE_QX=()          # qx range in keV (e.g., ("-5" "5")).
PLOT_RANGE_QY=()          # qy range in keV (e.g., ("-5" "5")).
PLOT_RANGE_QZ=()          # qz range in keV (e.g., ("-5" "5")).
PLOT_RANGE_X=()           # x range in Angstroms (e.g., ("-10" "10")).
PLOT_RANGE_Y=()           # y range in Angstroms (e.g., ("-10" "10")).
PLOT_RANGE_Z=()           # z range in Angstroms (e.g., ("-10" "10")).

# ========================================
# CONFIG ENDS HERE
# ========================================

# Get the directory where this script is located.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OSC_CHECK_DIR="$PROJECT_ROOT/src/form_factors/common"

# Use the Python venv and the Julia project at the repo root.
PYTHON_BIN="$PROJECT_ROOT/env/bin/python"
JULIA_BIN="julia --project=$PROJECT_ROOT"

TRANSITIONS_TO_PLOT=()
FIRST_TRANSITION=""

# Function to format seconds into human-readable time.
format_time() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    if [ $hours -gt 0 ]; then
        printf "%dh %dm %ds" $hours $minutes $seconds
    elif [ $minutes -gt 0 ]; then
        printf "%dm %ds" $minutes $seconds
    else
        printf "%ds" $seconds
    fi
}

lowercase() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

CARTESIAN_COMPUTE_MODE_LC="$(lowercase "$CARTESIAN_COMPUTE_MODE")"
TRANSITION_INDICES_LC="$(lowercase "$TRANSITION_INDICES")"

CRYSTAL_MODE=false
if [ -n "$CIF_FILE" ] || [ -n "$CIF_DIR" ]; then
    CRYSTAL_MODE=true
fi

# Remove any CUDA entries from paths so that julia stops complaining.
strip_cuda_from_path_var() {
    local var_name=$1
    local value="${!var_name:-}"
    local sanitised=""
    local IFS=':'

    for entry in $value; do
        if [[ "$entry" == *cuda* ]] || [[ "$entry" == *CUDA* ]]; then
            continue
        fi
        sanitised="${sanitised:+$sanitised:}$entry"
    done

    eval "$var_name=\"$sanitised\""
}

echo "=================================="
echo "Molecular form factor computation"
echo "=================================="
echo "Run name: $RUN_NAME"
echo "Output directory: $PROJECT_ROOT/runs/$RUN_NAME"
if [ "$CRYSTAL_MODE" = true ] && [ -n "$CIF_DIR" ]; then
    echo "Running calculation for the list of crystals found in ($CIF_DIR)."
elif [ "$CRYSTAL_MODE" = true ]; then
    echo "Running calculation for a single crystal from CIF ($CIF_FILE)."
elif [ -n "$CSV_FILE" ]; then
    echo "Running calculation for the list of molecules found in ($CSV_FILE)."
else
    echo "Running calculation for a single molecule with SMILES ($SMILES)."
fi
echo ""

# Step 1: Run the TD-DFT calculation.
echo "Step 1: Running the TD-DFT calculation..."

# Track the time taken for the TD-DFT calculation.
TDDFT_TIME=0

# Determine the precision suffix for filenames.
if [ "$PRECISION" = "float32" ]; then
    PRECISION_SUFFIX="_f32"
else
    PRECISION_SUFFIX="_f64"
fi

# Check if we should skip the computation.
RUN_DIR="$PROJECT_ROOT/runs/$RUN_NAME"
if [ "$SKIP_TDDFT" = true ] && [ -f "$RUN_DIR/metadata.csv" ]; then
    echo "  Skipping TD-DFT calculation and using existing results."
    echo ""
else
    echo "  Basis: $BASIS"
    echo "  XC functional: $XC_FUNCTIONAL"
    echo "  Number of states: $NSTATES"
    echo "  Number of transitions: $NTRANS"
    echo ""

    cd "$SCRIPT_DIR"

    # Build the td_dft command.
    TD_DFT_CMD="$PYTHON_BIN td_dft.py --basis \"$BASIS\" --xc \"$XC_FUNCTIONAL\" --nstates $NSTATES --ntrans $NTRANS --precision $PRECISION --output-dir \"$RUN_DIR\""

    # Add the input to the command.
    if [ "$CRYSTAL_MODE" = true ] && [ -n "$CIF_DIR" ]; then
        TD_DFT_CMD="$TD_DFT_CMD --cif-dir \"$CIF_DIR\""
    elif [ "$CRYSTAL_MODE" = true ]; then
        TD_DFT_CMD="$TD_DFT_CMD --cif-file \"$CIF_FILE\""
    elif [ -n "$CSV_FILE" ]; then
        TD_DFT_CMD="$TD_DFT_CMD --csv-file \"$CSV_FILE\""
    else
        TD_DFT_CMD="$TD_DFT_CMD --smiles \"$SMILES\""
    fi

    if [ -n "$RING_FLATTEN" ] && [ "$CRYSTAL_MODE" != true ]; then
        TD_DFT_CMD="$TD_DFT_CMD $RING_FLATTEN"
    fi
    if [ "$USE_GPU" = true ]; then
        TD_DFT_CMD="$TD_DFT_CMD --use-gpu"
    fi
    if [ "$DFT_OPTIMISATION" = true ] && [ "$CRYSTAL_MODE" != true ]; then
        TD_DFT_CMD="$TD_DFT_CMD --dft-optimisation"
    fi
    if [ "$PLOT_MOLECULE_3D" = true ] && [ "$CRYSTAL_MODE" != true ]; then
        TD_DFT_CMD="$TD_DFT_CMD --plot-molecule-3d"
    fi

    eval $TD_DFT_CMD

    # Read the computation time from the file written by Python.
    TDDFT_TIME_FILE="$RUN_DIR/.tddft_time"
    if [ -f "$TDDFT_TIME_FILE" ]; then
        TDDFT_TIME=$(awk '{print int($1 + 0.5)}' "$TDDFT_TIME_FILE")
        rm "$TDDFT_TIME_FILE"
    else
        TDDFT_TIME=0
    fi

    echo ""
    echo "TD-DFT calculation complete!"
    echo ""
fi

# Step 2: Compute the form factor.
echo "Step 2: Form factor computation..."

# Remove any CUDA entries from PATH and LD_LIBRARY_PATH to avoid Julia warnings.
strip_cuda_from_path_var PATH
strip_cuda_from_path_var LD_LIBRARY_PATH

# Track the time taken for the form factor computation.
FORM_FACTOR_TIME=0

# Check if form factor results already exist.
# We do this by looking for the expected output directory in molecule 1's folder.
FORM_FACTOR_EXISTS=false
if [ "$CRYSTAL_MODE" = true ]; then
    if [ -d "$RUN_DIR/1/crystal" ]; then
        FORM_FACTOR_EXISTS=true
    fi
elif [ -d "$RUN_DIR/1/$METHOD" ]; then
    FORM_FACTOR_EXISTS=true
fi

if [ "$SKIP_FORM_FACTOR" = true ] && [ "$FORM_FACTOR_EXISTS" = true ]; then
    echo "  Skipping form factor computation and using existing results."
    echo ""
else
    echo "  Method: $METHOD"
    echo "  Julia threads: $JULIA_THREADS"
    echo "  Precision: $PRECISION"
    echo "  Transition indices: $TRANSITION_INDICES"

    cd "$SCRIPT_DIR"

    # Build the Julia command.
    JULIA_CMD="$JULIA_BIN -t $JULIA_THREADS compute_form_factor.jl --output-dir \"$RUN_DIR\""

    # Add the input to the command.
    if [ "$CRYSTAL_MODE" = true ] && [ -n "$CIF_DIR" ]; then
        JULIA_CMD="$JULIA_CMD --cif-dir \"$CIF_DIR\" --crystal-mode"
    elif [ "$CRYSTAL_MODE" = true ]; then
        JULIA_CMD="$JULIA_CMD --cif-file \"$CIF_FILE\" --crystal-mode"
    elif [ -n "$CSV_FILE" ]; then
        JULIA_CMD="$JULIA_CMD --csv-file \"$CSV_FILE\""
    else
        JULIA_CMD="$JULIA_CMD --smiles \"$SMILES\""
    fi

    JULIA_CMD="$JULIA_CMD --method $METHOD"
    JULIA_CMD="$JULIA_CMD --transition-indices \"$TRANSITION_INDICES\""
    JULIA_CMD="$JULIA_CMD --precision $PRECISION"

    # Add method-specific parameters.
    if [ "$METHOD" = "spherical" ]; then
        echo "  l_max=$L_MAX"
        echo "  threshold=$THRESHOLD"
        echo ""
        JULIA_CMD="$JULIA_CMD --q-max $Q_MAX"
        JULIA_CMD="$JULIA_CMD --N-q $N_Q"
        JULIA_CMD="$JULIA_CMD --N-theta $N_THETA"
        JULIA_CMD="$JULIA_CMD --N-phi $N_PHI"
        JULIA_CMD="$JULIA_CMD --l-max $L_MAX"
        JULIA_CMD="$JULIA_CMD --threshold $THRESHOLD"
        COMPUTE_MODE_ARG=$(IFS=','; echo "${COMPUTE_MODES[*]}")
        JULIA_CMD="$JULIA_CMD --compute-mode \"$COMPUTE_MODE_ARG\""

        if [ "$COMPUTE_RATES" = true ]; then
            JULIA_CMD="$JULIA_CMD --compute-rates"
            JULIA_CMD="$JULIA_CMD --m-grid $M_GRID"
            JULIA_CMD="$JULIA_CMD --N-rotations $N_ROTATIONS"
        fi

        if [ "$FORCE_RECOMPUTATION" = true ]; then
            JULIA_CMD="$JULIA_CMD --force-recomputation"
        fi

        if [ "$USE_GPU" = true ]; then
            JULIA_CMD="$JULIA_CMD --use-gpu"
        fi

    elif [ "$METHOD" = "fft" ]; then
        echo "  q_lim=$Q_LIM keV"
        echo "  q_res=$Q_RES keV"
        echo ""
        JULIA_CMD="$JULIA_CMD --q-lim $Q_LIM"
        JULIA_CMD="$JULIA_CMD --q-res $Q_RES"

        if [ "$CHECK_PARSEVAL" = true ]; then
            JULIA_CMD="$JULIA_CMD --check-parseval"
        fi

        if [ "$USE_GPU" = true ]; then
            JULIA_CMD="$JULIA_CMD --use-gpu"
        fi


    elif [ "$METHOD" = "cartesian" ]; then
        echo "  qx_grid=$QX_GRID keV"
        echo "  qy_grid=$QY_GRID keV"
        echo "  qz_grid=$QZ_GRID keV"
        echo "  threshold=$THRESHOLD"
        echo "  compute_mode=$CARTESIAN_COMPUTE_MODE"
        echo ""
        JULIA_CMD="$JULIA_CMD --qx-grid $QX_GRID"
        JULIA_CMD="$JULIA_CMD --qy-grid $QY_GRID"
        JULIA_CMD="$JULIA_CMD --qz-grid $QZ_GRID"
        JULIA_CMD="$JULIA_CMD --threshold $THRESHOLD"
        JULIA_CMD="$JULIA_CMD --compute-mode $CARTESIAN_COMPUTE_MODE"

        if [ "$USE_GPU" = true ]; then
            JULIA_CMD="$JULIA_CMD --use-gpu"
        fi
    fi

    if [ "$BENCHMARK" = true ]; then
        JULIA_CMD="$JULIA_CMD --benchmark"
    fi

    eval $JULIA_CMD

    # Read the computation time from the file written by Julia.
    FORM_FACTOR_TIME_FILE="$RUN_DIR/.form_factor_time"
    if [ -f "$FORM_FACTOR_TIME_FILE" ]; then
        FORM_FACTOR_TIME=$(awk '{print int($1 + 0.5)}' "$FORM_FACTOR_TIME_FILE")
        rm "$FORM_FACTOR_TIME_FILE"
    else
        FORM_FACTOR_TIME=0
    fi

    echo ""
    echo "Form factor computation complete!"
    echo ""
fi

# Step 3: Generate plots.
echo "Step 3: Plot generation..."

# Check if both plot types are skipped, or if only R/V tensors were computed.
if [ "$SKIP_2D_PLOTS" = true ] && [ "$SKIP_3D_PLOTS" = true ]; then
    echo "  Skipping all plot generation."
    echo ""
elif [ "$METHOD" = "spherical" ] && ! printf '%s\n' "${COMPUTE_MODES[@]}" | grep -qi "form_factor" && ! { printf '%s\n' "${COMPUTE_MODES[@]}" | grep -qi "f_lm_tensor" && [ "$PLOT_FLM_MODES" = true ]; }; then
    echo "  Skipping plot generation (no form factor or f_lm tensor computed)."
    echo ""
elif [ "$METHOD" = "cartesian" ] && [ "$CARTESIAN_COMPUTE_MODE_LC" = "v_only" ]; then
    echo "  Skipping plot generation (only V tensors were computed, no form factor grid available)."
    echo ""
else
    cd "$SCRIPT_DIR"

    run_slice_plots() {
        local mol_num="$1"
        local results_rel_dir="$2"
        local include_rates="$3"
        local plot_cmd="$PYTHON_BIN plot_slices.py --run-name \"$RUN_NAME\" --molecule-number $mol_num --method $METHOD --results-dir \"$results_rel_dir\""

        if [ ${#TRANSITIONS_TO_PLOT[@]} -gt 0 ]; then
            local transition_arg
            transition_arg=$(IFS=','; echo "${TRANSITIONS_TO_PLOT[*]// /}")
            plot_cmd="$plot_cmd --transition-indices $transition_arg"
        fi

        if [ ${#PLOT_PLANES[@]} -gt 0 ]; then
            plot_cmd="$plot_cmd --planes ${PLOT_PLANES[@]}"
        fi

        if [ ${#PLOT_MODES[@]} -gt 0 ]; then
            plot_cmd="$plot_cmd --modes ${PLOT_MODES[@]}"
        fi

        if [ "$PLOT_TRANSITION_DENSITY" = true ]; then
            plot_cmd="$plot_cmd --plot-transition-density"
        fi

        if [ "$PLOT_FLM_MODES" = true ]; then
            plot_cmd="$plot_cmd --plot-flm-modes"
        fi

        if [ "$include_rates" = true ] && [ "$METHOD" = "spherical" ] && [ "$COMPUTE_RATES" = true ]; then
            plot_cmd="$plot_cmd --plot-rates"
        fi

        if [ ${#PLOT_RANGE_QX[@]} -gt 0 ]; then
            plot_cmd="$plot_cmd --qx-range=${PLOT_RANGE_QX[0]},${PLOT_RANGE_QX[1]}"
        fi
        if [ ${#PLOT_RANGE_QY[@]} -gt 0 ]; then
            plot_cmd="$plot_cmd --qy-range=${PLOT_RANGE_QY[0]},${PLOT_RANGE_QY[1]}"
        fi
        if [ ${#PLOT_RANGE_QZ[@]} -gt 0 ]; then
            plot_cmd="$plot_cmd --qz-range=${PLOT_RANGE_QZ[0]},${PLOT_RANGE_QZ[1]}"
        fi
        if [ ${#PLOT_RANGE_X[@]} -gt 0 ]; then
            plot_cmd="$plot_cmd --x-range=${PLOT_RANGE_X[0]},${PLOT_RANGE_X[1]}"
        fi
        if [ ${#PLOT_RANGE_Y[@]} -gt 0 ]; then
            plot_cmd="$plot_cmd --y-range=${PLOT_RANGE_Y[0]},${PLOT_RANGE_Y[1]}"
        fi
        if [ ${#PLOT_RANGE_Z[@]} -gt 0 ]; then
            plot_cmd="$plot_cmd --z-range=${PLOT_RANGE_Z[0]},${PLOT_RANGE_Z[1]}"
        fi

        eval $plot_cmd
    }

    run_3d_plots() {
        local mol_num="$1"
        local results_rel_dir="$2"

        for MODE in "${PLOT_3D_MODES[@]}"; do
            local plot_3d_cmd="$PYTHON_BIN plot_3d.py --run-name \"$RUN_NAME\" --molecule-number $mol_num --method $METHOD --results-dir \"$results_rel_dir\""
            plot_3d_cmd="$plot_3d_cmd --mode $MODE"
            if [ ${#TRANSITIONS_TO_PLOT[@]} -gt 0 ]; then
                local transition_arg
                transition_arg=$(IFS=','; echo "${TRANSITIONS_TO_PLOT[*]// /}")
                plot_3d_cmd="$plot_3d_cmd --transition-indices $transition_arg"
            fi
            plot_3d_cmd="$plot_3d_cmd --min-fraction $PLOT_3D_MIN_FRACTION"
            plot_3d_cmd="$plot_3d_cmd --max-fraction $PLOT_3D_MAX_FRACTION"
            plot_3d_cmd="$plot_3d_cmd --grid-size $PLOT_3D_GRID_SIZE"

            if [ "$PLOT_TRANSITION_DENSITY" = true ]; then
                plot_3d_cmd="$plot_3d_cmd --plot-transition-density"
            fi

            if [ ${#PLOT_RANGE_QX[@]} -gt 0 ]; then
                plot_3d_cmd="$plot_3d_cmd --qx-range=${PLOT_RANGE_QX[0]},${PLOT_RANGE_QX[1]}"
            fi
            if [ ${#PLOT_RANGE_QY[@]} -gt 0 ]; then
                plot_3d_cmd="$plot_3d_cmd --qy-range=${PLOT_RANGE_QY[0]},${PLOT_RANGE_QY[1]}"
            fi
            if [ ${#PLOT_RANGE_QZ[@]} -gt 0 ]; then
                plot_3d_cmd="$plot_3d_cmd --qz-range=${PLOT_RANGE_QZ[0]},${PLOT_RANGE_QZ[1]}"
            fi
            if [ ${#PLOT_RANGE_X[@]} -gt 0 ]; then
                plot_3d_cmd="$plot_3d_cmd --x-range=${PLOT_RANGE_X[0]},${PLOT_RANGE_X[1]}"
            fi
            if [ ${#PLOT_RANGE_Y[@]} -gt 0 ]; then
                plot_3d_cmd="$plot_3d_cmd --y-range=${PLOT_RANGE_Y[0]},${PLOT_RANGE_Y[1]}"
            fi
            if [ ${#PLOT_RANGE_Z[@]} -gt 0 ]; then
                plot_3d_cmd="$plot_3d_cmd --z-range=${PLOT_RANGE_Z[0]},${PLOT_RANGE_Z[1]}"
            fi

            eval $plot_3d_cmd
        done
    }

    # Get the number of molecules from metadata.csv
    METADATA_FILE="$RUN_DIR/metadata.csv"
    NUM_MOLECULES=$(tail -n +2 "$METADATA_FILE" | grep -c .)
    echo "  Plotting results for $NUM_MOLECULES molecule(s)..."
    echo ""

    # Loop over each molecule
    for MOL_NUM in $(seq 1 $NUM_MOLECULES); do
        MOL_DIR="$RUN_DIR/$MOL_NUM"
        if [ "$CRYSTAL_MODE" = true ]; then
            RESULTS_DIR="$MOL_DIR/crystal"
        else
            RESULTS_DIR="$MOL_DIR/$METHOD"
        fi

        if [ $NUM_MOLECULES -gt 1 ]; then
            echo "  --- Molecule $MOL_NUM ---"
        fi

        # Skip molecules that have no form factor results (e.g. TD-DFT failed).
        if [ ! -d "$RESULTS_DIR" ]; then
            echo "  No form factor results found for molecule $MOL_NUM, skipping."
            echo ""
            continue
        fi

        # Determine which transitions to plot.
        TRANSITIONS_TO_PLOT=()
        if [ "$TRANSITION_INDICES_LC" = "all" ]; then
            for dir in "$RESULTS_DIR"/*; do
                [ -d "$dir" ] || continue
                tidx=$(basename "$dir")
                [[ "$tidx" =~ ^[0-9]+$ ]] && TRANSITIONS_TO_PLOT+=("$tidx")
            done
        else
            IFS=',' read -ra TRANSITIONS_TO_PLOT <<< "$TRANSITION_INDICES"
        fi
        if [ ${#TRANSITIONS_TO_PLOT[@]} -gt 0 ]; then
            FIRST_TRANSITION="${TRANSITIONS_TO_PLOT[0]}"
        fi

        # Generate 2D slice plots.
        if [ "$SKIP_2D_PLOTS" = true ]; then
            if [ $NUM_MOLECULES -eq 1 ]; then
                echo "  Skipping 2D slice plot generation."
            fi
        else
            if [ $NUM_MOLECULES -eq 1 ]; then
                echo "  Generating 2D slice plots..."
            fi

            if [ "$CRYSTAL_MODE" = true ]; then
                run_slice_plots "$MOL_NUM" "crystal" true
                for conformer_dir in "$MOL_DIR"/conformers/*; do
                    [ -d "$conformer_dir" ] || continue
                    conformer_label=$(basename "$conformer_dir")
                    run_slice_plots "$MOL_NUM" "conformers/$conformer_label" true
                done
            else
                run_slice_plots "$MOL_NUM" "$METHOD" true
            fi

            if [ $NUM_MOLECULES -eq 1 ]; then
                echo "  2D slice plots complete!"
            fi
        fi

        # Generate 3D isosurface plots.
        if [ "$SKIP_3D_PLOTS" = true ]; then
            if [ $NUM_MOLECULES -eq 1 ]; then
                echo "  Skipping 3D isosurface plot generation."
            fi
        else
            if [ $NUM_MOLECULES -eq 1 ]; then
                echo "  Generating 3D isosurface plots..."
            fi

            if [ "$CRYSTAL_MODE" = true ]; then
                if [ $NUM_MOLECULES -eq 1 ]; then
                    echo "  Skipping 3D isosurface plot generation for crystal aggregate outputs."
                fi
                for conformer_dir in "$MOL_DIR"/conformers/*; do
                    [ -d "$conformer_dir" ] || continue
                    conformer_label=$(basename "$conformer_dir")
                    run_3d_plots "$MOL_NUM" "conformers/$conformer_label"
                done
            else
                run_3d_plots "$MOL_NUM" "$METHOD"
            fi

            if [ $NUM_MOLECULES -eq 1 ]; then
                echo ""
                echo "  3D isosurface plots complete!"
            fi
        fi

        if [ $NUM_MOLECULES -gt 1 ]; then
            echo ""
        fi
    done

    echo ""
    echo "Plot generation complete!"
    echo ""
fi

# Ensure that the transition list is populated for any downstream steps.
if [ ${#TRANSITIONS_TO_PLOT[@]} -eq 0 ]; then
    if [ "$TRANSITION_INDICES_LC" = "all" ]; then
        # Use molecule 1's directory to find the transitions.
        if [ "$CRYSTAL_MODE" = true ]; then
            RESULTS_DIR="$RUN_DIR/1/crystal"
        else
            RESULTS_DIR="$RUN_DIR/1/$METHOD"
        fi
        for dir in "$RESULTS_DIR"/*; do
            [ -d "$dir" ] || continue
            tidx=$(basename "$dir")
            [[ "$tidx" =~ ^[0-9]+$ ]] && TRANSITIONS_TO_PLOT+=("$tidx")
        done
    else
        IFS=',' read -ra TRANSITIONS_TO_PLOT <<< "$TRANSITION_INDICES"
    fi
    if [ ${#TRANSITIONS_TO_PLOT[@]} -gt 0 ]; then
        FIRST_TRANSITION="${TRANSITIONS_TO_PLOT[0]}"
    fi
fi

# Step 4: Verification.
if [ "$CHECK_OSCILLATOR_STRENGTH" = true ] && [ "$CRYSTAL_MODE" != true ]; then
    echo "Step 4: Verifying results..."
    echo ""

    cd "$OSC_CHECK_DIR"

    CHECK_OSC_CMD="$JULIA_BIN -t $JULIA_THREADS CheckOscillatorStrength.jl --smiles \"$SMILES\" --run-name \"$RUN_NAME\" --precision $PRECISION --method $METHOD --transition-indices \"$TRANSITION_INDICES\" --q-max-fit $Q_MAX_FIT"
    eval $CHECK_OSC_CMD

    echo ""
    echo "Verification complete!"
    echo ""
fi

echo "=================================="
echo "Form factor analysis complete!"
echo "=================================="
echo "Results saved to: $RUN_DIR/"
echo ""
echo "Directory structure:"
echo "  $RUN_NAME/"
echo "    ├── metadata.csv"
echo "    ├── 1/  (molecule 1)"

# Only report what was actually computed.
if [ $TDDFT_TIME -gt 0 ]; then
    if [ "$CRYSTAL_MODE" = true ]; then
        echo "    │   ├── crystal_metadata.json"
        echo "    │   └── conformers/"
        echo "    │       └── A/"
        echo "    │           └── td_dft_results${PRECISION_SUFFIX}.h5"
    else
        echo "    │   ├── td_dft_results${PRECISION_SUFFIX}.h5"
    fi
fi

if [ $FORM_FACTOR_TIME -gt 0 ]; then
    if [ "$CRYSTAL_MODE" = true ]; then
        echo "    │   ├── conformers/"
        echo "    │   │   └── A/"
        echo "    │   │       ├── 1/"
        echo "    │   │       │   └── fs_grid${PRECISION_SUFFIX}.h5"
        if [ "$METHOD" = "spherical" ] && [ "$COMPUTE_RATES" = true ]; then
            echo "    │   │       └── scattering_rates${PRECISION_SUFFIX}.h5"
        fi
        echo "    │   └── crystal/"
        echo "    │       ├── 1/"
        echo "    │       │   └── fs_grid${PRECISION_SUFFIX}.h5"
        if [ "$METHOD" = "spherical" ] && [ "$COMPUTE_RATES" = true ]; then
            echo "    │       └── scattering_rates${PRECISION_SUFFIX}.h5"
        fi
    else
        echo "    │   └── $METHOD/"
        echo "    │       └── transition_*/"
        echo "    │           └── fs_grid${PRECISION_SUFFIX}.h5"
    fi
fi

if [ "$SKIP_2D_PLOTS" != true ] && ! { [ "$METHOD" = "spherical" ] && ! printf '%s\n' "${COMPUTE_MODES[@]}" | grep -qi "form_factor" && ! { printf '%s\n' "${COMPUTE_MODES[@]}" | grep -qi "f_lm_tensor" && [ "$PLOT_FLM_MODES" = true ]; }; } && ! { [ "$METHOD" = "cartesian" ] && [ "$CARTESIAN_COMPUTE_MODE_LC" = "v_only" ]; }; then
    if [ "$CRYSTAL_MODE" = true ]; then
        echo "    │   │       └── 1/"
        if printf '%s\n' "${COMPUTE_MODES[@]}" | grep -qi "form_factor"; then
            echo "    │   │           └── form_factor_*.png"
        fi
        if [ "$PLOT_FLM_MODES" = true ]; then
            echo "    │   │           └── flm_modes.png"
        fi
        if [ "$METHOD" = "spherical" ] && [ "$COMPUTE_RATES" = true ]; then
            echo "    │   │       └── scattering_rates.png"
        fi
        if [ "$PLOT_FLM_MODES" = true ]; then
            echo "    │       └── 1/"
            echo "    │           └── flm_modes.png"
        fi
        if [ "$METHOD" = "spherical" ] && [ "$COMPUTE_RATES" = true ]; then
            echo "    │       └── scattering_rates.png"
        fi
    else
        echo "    │           └── form_factor_*.png"
    fi
fi

if [ "$SKIP_3D_PLOTS" != true ] && ! { [ "$METHOD" = "spherical" ] && ! printf '%s\n' "${COMPUTE_MODES[@]}" | grep -qi "form_factor"; } && ! { [ "$METHOD" = "cartesian" ] && [ "$CARTESIAN_COMPUTE_MODE_LC" = "v_only" ]; }; then
    if [ "$CRYSTAL_MODE" = true ]; then
        echo "    │   │       └── 1/"
        echo "    │   │           └── form_factor_3d_*.html"
    else
        echo "    │           └── form_factor_3d_*.html"
    fi
fi

echo "    ├── 2/  (molecule 2, if applicable)"
echo "    └── ..."
echo ""

echo ""

# Show computation times only for what was actually computed.
TOTAL_TIME=$((TDDFT_TIME + FORM_FACTOR_TIME))
if [ $TOTAL_TIME -gt 0 ]; then
    echo "Computation times:"
    if [ $TDDFT_TIME -gt 0 ]; then
        echo "  TD-DFT:       $(format_time $TDDFT_TIME)"
    fi
    if [ $FORM_FACTOR_TIME -gt 0 ]; then
        echo "  Form factor:  $(format_time $FORM_FACTOR_TIME)"
    fi
    echo "  Total:        $(format_time $TOTAL_TIME)"
fi

echo "=================================="
