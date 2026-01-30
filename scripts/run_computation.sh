#!/bin/bash
# Pipeline script to run TD-DFT, compute form factors, and generate plots.

set -e  # Exit if anything breaks.

# ========================================
# CONFIG STARTS HERE
# ========================================

# Run configuration.
RUN_NAME="benzene_spherical"      # Name for this run.
CSV_FILE=""                 # Path to CSV file containing SMILES strings. Leave this empty for a single molecule run.

# Molecule specification.
SMILES="C1=CC=CC=C1" # SMILES string for the molecule. This is ignored if CSV_FILE is specified.

# ==== TD-DFT parameters. ====
BASIS="6-31g*"             # Basis set to use. Currently only "6-31g*" and "ccpvdz" are supported.
NSTATES=12                  # Number of excited states to compute.
NTRANS=12                   # Number of transitions to analyse.
RING_FLATTEN="--no-ring-flatten"            # Set to "--no-ring-flatten" to flatten based on whole molecule, or leave as "" for ring-based flattening.
DFT_OPTIMISATION=false    # Set to true to perform a DFT geometry optimisation after RDKit. Can be very slow on the CPU for large molecules.
PLOT_MOLECULE_3D=false    # Set to true to generate interactive 3D molecule plot from TD-DFT.

# ==== Form factor computation parameters. ====
METHOD="spherical"         # Method for form factor computation. Options: "spherical", "fft", or "cartesian".

# ==== Spherical method parameters. ====
# Momentum transfer grid parameters.
Q_MAX=10.0                 # Maximum momentum transfer in keV.
N_Q=201                    # Number of |q| grid points.
N_THETA=201                # Number of theta (polar angle) grid points.
N_PHI=201                  # Number of phi (azimuthal angle) grid points.

# Spherical harmonic expansion parameters.
L_MAX=24                   # Maximum angular mode, l, to include in spherical harmonic expansion.
COMPUTE_MODE="form_factor"        # What to compute/save for the spherical method. Options: "R_only", "form_factor", "both". If "R_only" is selected, the computation stops before the spherical grid contraction, and only R is saved to disk.

# ==== FFT method parameters. ====
Q_LIM="25.0,25.0,25.0"           # q-space limits in keV, comma-separated (qx_max, qy_max, qz_max).
Q_RES="0.125,0.125,0.125"        # q-space resolution in keV, comma-separated (Δqx, Δqy, Δqz).
CHECK_PARSEVAL=false       # Set to true to check Parseval's theorem holds.

# ==== Cartesian method parameters. ====
QX_GRID="-15,15,51"       # qx grid specification (min,max,N) in keV.
QY_GRID="-15,15,51"       # qy grid specification (min,max,N) in keV.
QZ_GRID="-15,15,51"       # qz grid specification (min,max,N) in keV.
CARTESIAN_COMPUTE_MODE="form_factor"  # What to compute/save for the Cartesian method. Options: "V_only", "form_factor", "both". If "V_only" is selected, the computation stops before the grid contraction, and only V tensors are saved to disk.

# ==== Computation parameters. ====
TRANSITION_INDICES="2"       # Which electronic transitions to compute. Can be "all" or a comma-separated list like "1,2,3,4". The first excited state is index 1.
THRESHOLD=1e-6             # Threshold for dropping small tensor values. For spherical: |W/W_max| < THRESHOLD. For Cartesian: |M_ij/M_max| < THRESHOLD. Set to 0.0 to disable. Around 1e-6 is recommended for both accuracy and performance.
JULIA_THREADS="auto"       # Number of threads to use for the Julia part of the code. Set to "auto" to use all available threads.
PRECISION="float32"       # Floating point precision for form factor computation. On CPU this has negligible impact on performance, but cuts down on file size by ~2. May produce 'fuzzy looking' Re/Im plots when the values are effectively zero in float32.
USE_GPU=false              # Set to true to enable GPU acceleration for the DFT and form factor computations. Supports spherical, FFT, and Cartesian methods.

# ==== Control flags. ====
SKIP_TDDFT=true            # Set to true to skip the TD-DFT calculation. Will not be skipped if the results do not already exist.
SKIP_FORM_FACTOR=true     # Set to true to skip the form factor computation. Will not be skipped if the results do not already exist.
SKIP_2D_PLOTS=false         # Set to true to skip 2D slice plot generation.
SKIP_3D_PLOTS=true         # Set to true to skip 3D isosurface plot generation.
FORCE_RECOMPUTATION=false  # Set to true to force recomputation of A and Gaunt coefficients in the form factor computation.
BENCHMARK=false            # Set to true to run benchmark after form factor computation. Testing only.

# ==== Verification. ====
CHECK_OSCILLATOR_STRENGTH=false  # Set to true to compute and compare oscillator strengths found from the form factor with those from PySCF.
Q_MAX_FIT=0.05                  # Maximum q value in keV for fitting the oscillator strength. You should ensure that at least 2 points sit below this value. It is highly recommended to set this to 0.2 keV or lower, the lower the better.

# ==== Plotting parameters. ====
USE_TEX=true              # Set to true to use TeX for text rendering in plots (requires TeX installation).
PLOT_TRANSITION_DENSITY=true  # Set to true to plot the transition density (FFT method only, will be ignored for the spherical method).

# 2D plotting parameters.
PLOT_PLANES=("xy" "xz" "yz")  # Planes to plot. Options: xy (qz=0), xz (qy=0), yz (qx=0). Each option will be plotted as a column, in the same order as provided.
PLOT_MODES=("modsq" "Re" "Im")  # What to plot. Options: modsq (|f_s|^2), Re (real part), Im (imaginary part). Each option will be plotted as row, in the same order as provided.

# 3D plotting parameters.
PLOT_3D_MODES=("modsq")  # Modes for 3D plots. Options: modsq (|f_S|^2), Re (real part), Im (imaginary part). Each mode will generate a separate 3D plot.
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
if [ -n "$CSV_FILE" ]; then
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
    echo "  Number of states: $NSTATES"
    echo "  Number of transitions: $NTRANS"
    echo ""

    cd "$SCRIPT_DIR"

    # Build the td_dft command.
    TD_DFT_CMD="$PYTHON_BIN td_dft.py --basis \"$BASIS\" --nstates $NSTATES --ntrans $NTRANS --precision $PRECISION --output-dir \"$RUN_DIR\""

    # Add the input to the command.
    if [ -n "$CSV_FILE" ]; then
        TD_DFT_CMD="$TD_DFT_CMD --csv-file \"$CSV_FILE\""
    else
        TD_DFT_CMD="$TD_DFT_CMD --smiles \"$SMILES\""
    fi

    if [ -n "$RING_FLATTEN" ]; then
        TD_DFT_CMD="$TD_DFT_CMD $RING_FLATTEN"
    fi
    if [ "$USE_GPU" = true ]; then
        TD_DFT_CMD="$TD_DFT_CMD --use-gpu"
    fi
    if [ "$DFT_OPTIMISATION" = true ]; then
        TD_DFT_CMD="$TD_DFT_CMD --dft-optimisation"
    fi
    if [ "$PLOT_MOLECULE_3D" = true ]; then
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
# We do this by looking for a method directory in molecule 1's folder.
FORM_FACTOR_EXISTS=false
if [ -d "$RUN_DIR/1/$METHOD" ]; then
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
    if [ -n "$CSV_FILE" ]; then
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
        JULIA_CMD="$JULIA_CMD --compute-mode $COMPUTE_MODE"

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
elif [ "$METHOD" = "spherical" ] && [ "${COMPUTE_MODE,,}" = "r_only" ]; then
    echo "  Skipping plot generation (only R tensor was computed, no form factor grid available)."
    echo ""
elif [ "$METHOD" = "cartesian" ] && [ "${CARTESIAN_COMPUTE_MODE,,}" = "v_only" ]; then
    echo "  Skipping plot generation (only V tensors were computed, no form factor grid available)."
    echo ""
else
    cd "$SCRIPT_DIR"

    # Get the number of molecules from metadata.csv
    METADATA_FILE="$RUN_DIR/metadata.csv"
    NUM_MOLECULES=$(tail -n +2 "$METADATA_FILE" | grep -c .)
    echo "  Plotting results for $NUM_MOLECULES molecule(s)..."
    echo ""

    # Loop over each molecule
    for MOL_NUM in $(seq 1 $NUM_MOLECULES); do
        MOL_DIR="$RUN_DIR/$MOL_NUM"

        if [ $NUM_MOLECULES -gt 1 ]; then
            echo "  --- Molecule $MOL_NUM ---"
        fi

        # Skip molecules that have no form factor results (e.g. TD-DFT failed).
        if [ ! -d "$MOL_DIR/$METHOD" ]; then
            echo "  No form factor results found for molecule $MOL_NUM, skipping."
            echo ""
            continue
        fi

        # Determine which transitions to plot.
        TRANSITIONS_TO_PLOT=()
        if [ "${TRANSITION_INDICES,,}" = "all" ]; then
            for dir in "$MOL_DIR/$METHOD"/transition_*; do
                [ -d "$dir" ] || continue
                tidx=$(basename "$dir" | cut -d'_' -f2)
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

            # Build the plot command for the 2D slice plots.
            PLOT_CMD="$PYTHON_BIN plot_slices.py --run-name \"$RUN_NAME\" --molecule-number $MOL_NUM --method $METHOD"
            if [ ${#TRANSITIONS_TO_PLOT[@]} -gt 0 ]; then
                TRANSITION_ARG=$(IFS=','; echo "${TRANSITIONS_TO_PLOT[*]// /}")
                PLOT_CMD="$PLOT_CMD --transition-indices $TRANSITION_ARG"
            fi

            # Add the planes to plot.
            if [ ${#PLOT_PLANES[@]} -gt 0 ]; then
                PLOT_CMD="$PLOT_CMD --planes ${PLOT_PLANES[@]}"
            fi

            # Add the modes to plot.
            if [ ${#PLOT_MODES[@]} -gt 0 ]; then
                PLOT_CMD="$PLOT_CMD --modes ${PLOT_MODES[@]}"
            fi

            if [ "$USE_TEX" = true ]; then
                PLOT_CMD="$PLOT_CMD --use-tex"
            fi

            if [ "$PLOT_TRANSITION_DENSITY" = true ]; then
                PLOT_CMD="$PLOT_CMD --plot-transition-density"
            fi

            # Add plot range limits if specified.
            if [ ${#PLOT_RANGE_QX[@]} -gt 0 ]; then
                PLOT_CMD="$PLOT_CMD --qx-range=${PLOT_RANGE_QX[0]},${PLOT_RANGE_QX[1]}"
            fi
            if [ ${#PLOT_RANGE_QY[@]} -gt 0 ]; then
                PLOT_CMD="$PLOT_CMD --qy-range=${PLOT_RANGE_QY[0]},${PLOT_RANGE_QY[1]}"
            fi
            if [ ${#PLOT_RANGE_QZ[@]} -gt 0 ]; then
                PLOT_CMD="$PLOT_CMD --qz-range=${PLOT_RANGE_QZ[0]},${PLOT_RANGE_QZ[1]}"
            fi
            if [ ${#PLOT_RANGE_X[@]} -gt 0 ]; then
                PLOT_CMD="$PLOT_CMD --x-range=${PLOT_RANGE_X[0]},${PLOT_RANGE_X[1]}"
            fi
            if [ ${#PLOT_RANGE_Y[@]} -gt 0 ]; then
                PLOT_CMD="$PLOT_CMD --y-range=${PLOT_RANGE_Y[0]},${PLOT_RANGE_Y[1]}"
            fi
            if [ ${#PLOT_RANGE_Z[@]} -gt 0 ]; then
                PLOT_CMD="$PLOT_CMD --z-range=${PLOT_RANGE_Z[0]},${PLOT_RANGE_Z[1]}"
            fi

            eval $PLOT_CMD

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

            # Generate a plot for each mode specified.
            for MODE in "${PLOT_3D_MODES[@]}"; do

                # Build the plot command for the 3D isosurface plot.
                PLOT_3D_CMD="$PYTHON_BIN plot_3d.py --run-name \"$RUN_NAME\" --molecule-number $MOL_NUM --method $METHOD"
                PLOT_3D_CMD="$PLOT_3D_CMD --mode $MODE"
                if [ ${#TRANSITIONS_TO_PLOT[@]} -gt 0 ]; then
                    TRANSITION_ARG=$(IFS=','; echo "${TRANSITIONS_TO_PLOT[*]// /}")
                    PLOT_3D_CMD="$PLOT_3D_CMD --transition-indices $TRANSITION_ARG"
                fi
                PLOT_3D_CMD="$PLOT_3D_CMD --min-fraction $PLOT_3D_MIN_FRACTION"
                PLOT_3D_CMD="$PLOT_3D_CMD --max-fraction $PLOT_3D_MAX_FRACTION"
                PLOT_3D_CMD="$PLOT_3D_CMD --grid-size $PLOT_3D_GRID_SIZE"

                if [ "$PLOT_TRANSITION_DENSITY" = true ]; then
                    PLOT_3D_CMD="$PLOT_3D_CMD --plot-transition-density"
                fi

                # Add the plot range limits if specified.
                if [ ${#PLOT_RANGE_QX[@]} -gt 0 ]; then
                    PLOT_3D_CMD="$PLOT_3D_CMD --qx-range=${PLOT_RANGE_QX[0]},${PLOT_RANGE_QX[1]}"
                fi
                if [ ${#PLOT_RANGE_QY[@]} -gt 0 ]; then
                    PLOT_3D_CMD="$PLOT_3D_CMD --qy-range=${PLOT_RANGE_QY[0]},${PLOT_RANGE_QY[1]}"
                fi
                if [ ${#PLOT_RANGE_QZ[@]} -gt 0 ]; then
                    PLOT_3D_CMD="$PLOT_3D_CMD --qz-range=${PLOT_RANGE_QZ[0]},${PLOT_RANGE_QZ[1]}"
                fi
                if [ ${#PLOT_RANGE_X[@]} -gt 0 ]; then
                    PLOT_3D_CMD="$PLOT_3D_CMD --x-range=${PLOT_RANGE_X[0]},${PLOT_RANGE_X[1]}"
                fi
                if [ ${#PLOT_RANGE_Y[@]} -gt 0 ]; then
                    PLOT_3D_CMD="$PLOT_3D_CMD --y-range=${PLOT_RANGE_Y[0]},${PLOT_RANGE_Y[1]}"
                fi
                if [ ${#PLOT_RANGE_Z[@]} -gt 0 ]; then
                    PLOT_3D_CMD="$PLOT_3D_CMD --z-range=${PLOT_RANGE_Z[0]},${PLOT_RANGE_Z[1]}"
                fi

                eval $PLOT_3D_CMD
            done

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
    if [ "${TRANSITION_INDICES,,}" = "all" ]; then
        # Use molecule 1's directory to find the transitions.
        for dir in "$RUN_DIR/1/$METHOD"/transition_*; do
            [ -d "$dir" ] || continue
            tidx=$(basename "$dir" | cut -d'_' -f2)
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
if [ "$CHECK_OSCILLATOR_STRENGTH" = true ]; then
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
    echo "    │   ├── td_dft_results${PRECISION_SUFFIX}.h5"
fi

if [ $FORM_FACTOR_TIME -gt 0 ]; then
    echo "    │   └── $METHOD/"
    echo "    │       └── transition_*/"
    echo "    │           └── fs_grid${PRECISION_SUFFIX}.h5"
fi

if [ "$SKIP_2D_PLOTS" != true ] && ! { [ "$METHOD" = "spherical" ] && [ "${COMPUTE_MODE,,}" = "r_only" ]; } && ! { [ "$METHOD" = "cartesian" ] && [ "${CARTESIAN_COMPUTE_MODE,,}" = "v_only" ]; }; then
    echo "    │           └── form_factor_*.png"
fi

if [ "$SKIP_3D_PLOTS" != true ] && ! { [ "$METHOD" = "spherical" ] && [ "${COMPUTE_MODE,,}" = "r_only" ]; } && ! { [ "$METHOD" = "cartesian" ] && [ "${CARTESIAN_COMPUTE_MODE,,}" = "v_only" ]; }; then
    echo "    │           └── form_factor_3d_*.html"
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
