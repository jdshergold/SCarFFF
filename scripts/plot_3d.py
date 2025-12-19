# This script creates 3D isosurface visualisations of form factors and transition densities.
# Works with both spherical and FFT form factor outputs (transition density is FFT only).

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator


def parse_cli_args():
    """
    Parse the command line arguments and return the parsed arguments.

    # Arguments:
    - None.

    # Returns:
    - parsed_args::argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create 3D isosurface visualisations of the form factor or transition density."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name of the run. The results will be loaded from runs/<run-name>/<molecule-number>/.",
    )
    parser.add_argument(
        "--molecule-number",
        type=int,
        required=True,
        help="Molecule number (e.g., 1, 2, 3...) within the run.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["spherical", "fft", "cartesian"],
        help="Form factor computation method. Must match the method used in compute_form_factor.jl.",
    )
    parser.add_argument(
        "--transition-indices",
        type=str,
        default="1",
        help="Transition indices to plot (spherical and Cartesian methods). Comma-separated, defaults to 1.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="modsq",
        choices=["modsq", "Re", "Im"],
        help="What to plot. The options are modsq (|f_S|^2), Im (Im(f_S)), Re (Re(f_S)). Default: modsq.",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.15,
        help="Minimum isosurface level as a fraction of maximum. Default: 0.15.",
    )
    parser.add_argument(
        "--max-fraction",
        type=float,
        default=0.95,
        help="Maximum isosurface level as a fraction of maximum. Setting to 1 is not recommended, as this will be a single point. Default: 0.95.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=75,
        help="Target grid (x,y,z) size for downsampling. Setting this too large will make the plots laggy and slow to load. Default: 75.",
    )
    parser.add_argument(
        "--qx-range",
        type=str,
        default=None,
        help="qx range to plot, given as 'min,max' in keV (e.g., '-5,5'). If not specified, the full data range will be plotted.",
    )
    parser.add_argument(
        "--qy-range",
        type=str,
        default=None,
        help="qy range to plot, given as 'min,max' in keV (e.g., '-5,5'). If not specified, the full data range will be plotted.",
    )
    parser.add_argument(
        "--qz-range",
        type=str,
        default=None,
        help="qz range to plot, given as 'min,max' in keV (e.g., '-5,5'). If not specified, the full data range will be plotted.",
    )
    parser.add_argument(
        "--plot-transition-density",
        action="store_true",
        help="Whether to plot the transition density (FFT method only).",
    )
    parser.add_argument(
        "--x-range",
        type=str,
        default=None,
        help="x range to plot for transition density, given as 'min,max' in Angstroms (e.g., '-10,10'). If not specified, the full data range will be plotted.",
    )
    parser.add_argument(
        "--y-range",
        type=str,
        default=None,
        help="y range to plot for transition density, given as 'min,max' in Angstroms (e.g., '-10,10'). If not specified, the full data range will be plotted.",
    )
    parser.add_argument(
        "--z-range",
        type=str,
        default=None,
        help="z range to plot for transition density, given as 'min,max' in Angstroms (e.g., '-10,10'). If not specified, the full data range will be plotted.",
    )

    parsed_args = parser.parse_args()

    # Parse range arguments.
    def parse_range(range_str):
        """
        Parse a range string into min and max values.

        # Arguments:
        - range_str::str or None: Range string in format 'min,max' (e.g., '-5,5'), or None.

        # Returns:
        - min_val::float or None: Minimum value, or None if range_str is None.
        - max_val::float or None: Maximum value, or None if range_str is None.
        """
        if range_str is None:
            return None, None
        parts = range_str.split(',')
        if len(parts) != 2:
            raise ValueError(f"Range must be 'min,max', got: {range_str}")
        return float(parts[0]), float(parts[1])
    
    parsed_args.qx_min, parsed_args.qx_max = parse_range(parsed_args.qx_range)
    parsed_args.qy_min, parsed_args.qy_max = parse_range(parsed_args.qy_range)
    parsed_args.qz_min, parsed_args.qz_max = parse_range(parsed_args.qz_range)
    parsed_args.x_min, parsed_args.x_max = parse_range(parsed_args.x_range)
    parsed_args.y_min, parsed_args.y_max = parse_range(parsed_args.y_range)
    parsed_args.z_min, parsed_args.z_max = parse_range(parsed_args.z_range)

    # Get the directory where this script is located.
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    # Construct the input and output paths based on run name and molecule number.
    runs_dir = project_root / "runs" / parsed_args.run_name / str(parsed_args.molecule_number)

    # Set the output directory.
    parsed_args.output_dir = runs_dir / parsed_args.method

    if parsed_args.method not in ["spherical", "fft", "cartesian"]:
        print(f"Error: Invalid method '{parsed_args.method}'. Must be 'spherical', 'fft', or 'cartesian'.")
        sys.exit(1)

    # The transition density is only available for the FFT method.
    if parsed_args.plot_transition_density and parsed_args.method != "fft":
        parsed_args.plot_transition_density = False

    return parsed_args


def interpolate_to_cartesian_grid(q_grid, theta_grid, phi_grid, data, grid_size, q_lim=None):
    """
    Interpolate spherical grid data onto a regular Cartesian grid.

    # Arguments:
    - q_grid::np.ndarray: The |q| grid.
    - theta_grid::np.ndarray: The theta grid.
    - phi_grid::np.ndarray: The phi grid.
    - data::np.ndarray: 3D array of values on the spherical grid.
    - grid_size::int: The target grid size for downsampling. This is the number of points in each direction.
    - q_lim::tuple or None: Optional (qx_max, qy_max, qz_max) limits. If None, uses q_grid.max().

    # Returns:
    - qx_mesh::np.ndarray: Uniform 3D grid of q_x coordinates.
    - qy_mesh::np.ndarray: Uniform 3D grid of q_y coordinates.
    - qz_mesh::np.ndarray: Uniform 3D grid of q_z coordinates.
    - grid_data::np.ndarray: The interpolated data on the uniform grid.
    """

    # Convert to float32 to save disk space in the final plot.
    data = np.asarray(data, dtype=np.float32)

    # Interpolate based on the spherical grid.
    interpolator = RegularGridInterpolator(
        (theta_grid, phi_grid, q_grid),
        data,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Determine the grid limits.
    if q_lim is None:
        q_max = q_grid.max()
        qx_max = qy_max = qz_max = q_max
    else:
        qx_max, qy_max, qz_max = q_lim

    # Generate the regular Cartesian grid, also in float32.
    qx_1d = np.linspace(-qx_max, qx_max, grid_size, dtype=np.float32)
    qy_1d = np.linspace(-qy_max, qy_max, grid_size, dtype=np.float32)
    qz_1d = np.linspace(-qz_max, qz_max, grid_size, dtype=np.float32)
    qx_mesh, qy_mesh, qz_mesh = np.meshgrid(qx_1d, qy_1d, qz_1d, indexing="ij")

    # Now convert the Cartesian grid to spherical coordinates for the interpolation.
    q_vals = np.sqrt(qx_mesh**2 + qy_mesh**2 + qz_mesh**2)
    theta_vals = np.arccos(
        np.clip(qz_mesh / (q_vals + 1e-10), -1, 1)
    )  # Add in a small offset to avoid division by zero at the origin, and restrict cos(theta) to [-1, 1].
    phi_vals = np.arctan2(qy_mesh, qx_mesh)
    phi_vals[phi_vals < 0] += (
        2 * np.pi
    )  # Match any negative phi to those in the original dataset, which are in [0, 2pi].

    # Interpolate and reshape. Cast to float32 as scipy interpolate doesn't necessarily preserve dtypes.
    grid_points = np.stack(
        [theta_vals.flatten(), phi_vals.flatten(), q_vals.flatten()], axis=-1
    )
    grid_data = (
        interpolator(grid_points)
        .reshape(grid_size, grid_size, grid_size)
        .astype(np.float32)
    )

    # Set values outside the original sphere to zero.
    mask = q_vals > q_grid.max()
    grid_data[mask] = 0.0

    return qx_mesh, qy_mesh, qz_mesh, grid_data


def downsample_fft_grid(qx_coords, qy_coords, qz_coords, data, grid_size):
    """
    Downsample the FFT Cartesian grid data to the target grid size.

    # Arguments:
    - qx_coords::np.ndarray: 1D array of qx coordinates.
    - qy_coords::np.ndarray: 1D array of qy coordinates.
    - qz_coords::np.ndarray: 1D array of qz coordinates.
    - data::np.ndarray: 3D array of values on the Cartesian grid.
    - grid_size::int: The target grid size for downsampling.

    # Returns:
    - qx_mesh::np.ndarray: Downsampled 3D grid of q_x coordinates.
    - qy_mesh::np.ndarray: Downsampled 3D grid of q_y coordinates.
    - qz_mesh::np.ndarray: Downsampled 3D grid of q_z coordinates.
    - grid_data::np.ndarray: The downsampled data on the uniform grid.
    """
    # Convert to float32 to save space.
    data = np.asarray(data, dtype=np.float32)

    # Create the interpolator for the FFT data.
    interpolator = RegularGridInterpolator(
        (qx_coords, qy_coords, qz_coords),
        data,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Generate the downsampled Cartesian grid.
    qx_1d = np.linspace(qx_coords[0], qx_coords[-1], grid_size, dtype=np.float32)
    qy_1d = np.linspace(qy_coords[0], qy_coords[-1], grid_size, dtype=np.float32)
    qz_1d = np.linspace(qz_coords[0], qz_coords[-1], grid_size, dtype=np.float32)
    qx_mesh, qy_mesh, qz_mesh = np.meshgrid(qx_1d, qy_1d, qz_1d, indexing="ij")

    # Interpolate to the downsampled grid.
    grid_points = np.stack(
        [qx_mesh.flatten(), qy_mesh.flatten(), qz_mesh.flatten()], axis=-1
    )
    grid_data = (
        interpolator(grid_points)
        .reshape(grid_size, grid_size, grid_size)
        .astype(np.float32)
    )

    return qx_mesh, qy_mesh, qz_mesh, grid_data


def apply_range_limits(coord1_mesh, coord2_mesh, coord3_mesh, data, args, coord_type):
    """
    Apply the range limits to the data for plotting by trimming to the requested region.
    
    # Arguments:
    - coord1_mesh::np.ndarray: 3D grid of first coordinate.
    - coord2_mesh::np.ndarray: 3D grid of second coordinate.
    - coord3_mesh::np.ndarray: 3D grid of third coordinate.
    - data::np.ndarray: 3D array of data values.
    - args::argparse.Namespace: Command line arguments.
    - coord_type::str: Type of coordinates, this is "q" for momentum space, or "r" for real space.
    
    # Returns:
    - coord1_mesh::np.ndarray: The trimmed first coordinates.
    - coord2_mesh::np.ndarray: The trimmed second coordinates.
    - coord3_mesh::np.ndarray: The trimmed third coordinates.
    - data::np.ndarray: The trimmed data.
    """
    # Determine the requested limits.
    if coord_type == "q":
        coord1_min, coord1_max = args.qx_min, args.qx_max
        coord2_min, coord2_max = args.qy_min, args.qy_max
        coord3_min, coord3_max = args.qz_min, args.qz_max
    else:
        coord1_min, coord1_max = args.x_min, args.x_max
        coord2_min, coord2_max = args.y_min, args.y_max
        coord3_min, coord3_max = args.z_min, args.z_max

    # If no limits are set, just return the inputs as-is.
    if all(
        limit is None
        for limit in [
            coord1_min,
            coord1_max,
            coord2_min,
            coord2_max,
            coord3_min,
            coord3_max,
        ]
    ):
        return coord1_mesh, coord2_mesh, coord3_mesh, data

    # The grid is uniform and monotonic along each axis, so we can find slices on the 1D coordinate arrays.
    coord1_vals = coord1_mesh[:, 0, 0]
    coord2_vals = coord2_mesh[0, :, 0]
    coord3_vals = coord3_mesh[0, 0, :]

    def get_slice(vals, min_val, max_val):
        mask = np.ones_like(vals, dtype=bool)
        if min_val is not None:
            mask &= vals >= min_val
        if max_val is not None:
            mask &= vals <= max_val
        if not mask.any():
            return None
        idx = np.where(mask)[0]
        return idx.min(), idx.max()

    idx1 = get_slice(coord1_vals, coord1_min, coord1_max)
    idx2 = get_slice(coord2_vals, coord2_min, coord2_max)
    idx3 = get_slice(coord3_vals, coord3_min, coord3_max)

    # Return an error if there are no points in the trimmed region.
    if idx1 is None or idx2 is None or idx3 is None:
        print("Error: No grid points remain within the specified plot ranges.")
        sys.exit(1)

    i1_min, i1_max = idx1
    i2_min, i2_max = idx2
    i3_min, i3_max = idx3

    coord1_trim = coord1_mesh[i1_min : i1_max + 1, i2_min : i2_max + 1, i3_min : i3_max + 1]
    coord2_trim = coord2_mesh[i1_min : i1_max + 1, i2_min : i2_max + 1, i3_min : i3_max + 1]
    coord3_trim = coord3_mesh[i1_min : i1_max + 1, i2_min : i2_max + 1, i3_min : i3_max + 1]
    data_trim = data[i1_min : i1_max + 1, i2_min : i2_max + 1, i3_min : i3_max + 1]

    return coord1_trim, coord2_trim, coord3_trim, data_trim


def extract_domain_data(data, mode, is_transition_density):
    """
    Get the correct data to plot based on the selected mode.

    # Arguments:
    - data::np.ndarray: Complex array of 3D form factor values, or real array of transition density values.
    - mode::str or None: What to plot for form factors. Should be one of "modsq", "Im", or "Re". This is ignored for transition density.
    - is_transition_density::bool: Whether we are plotting the transition density.

    # Returns:
    - plot_data::np.ndarray: The data to plot.
    - label::str: The corresponding colourbar label.
    - colorscale::str: The colourmap to use.
    - symmetric::bool: Whether to use symmetric colourbar limits.
    """
    if is_transition_density:
        plot_data = data
        label = "ρ(r) [Å<sup>-3</sup>]"
        colorscale = "RdBu_r"
        symmetric = True
    else:
        if mode == "modsq":
            plot_data = np.abs(data) ** 2
            label = "|f_S(q)|^2"
            colorscale = "Viridis"
            symmetric = False
        elif mode == "Im":
            plot_data = np.imag(data)
            label = "Im[f_S(q)]"
            colorscale = "RdBu_r"
            symmetric = True
        elif mode == "Re":
            plot_data = np.real(data)
            label = "Re[f_S(q)]"
            colorscale = "RdBu_r"
            symmetric = True
        else:
            raise ValueError(
                f"Invalid mode {mode} selected. Must be one of 'modsq', 'Im', or 'Re'."
            )

    return plot_data, label, colorscale, symmetric


def create_isosurface_plot(coord1_mesh, coord2_mesh, coord3_mesh, data, label, colorscale, symmetric, title, axis_labels, min_fraction, max_fraction):
    """
    Create a 3D isosurface plot using plotly Isosurface.

    # Arguments:
    - coord1_mesh::np.ndarray: Uniform 3D grid of first coordinates.
    - coord2_mesh::np.ndarray: Uniform 3D grid of second coordinates.
    - coord3_mesh::np.ndarray: Uniform 3D grid of third coordinates.
    - data::np.ndarray: 3D array of values to plot.
    - label::str: Colourbar label.
    - colorscale::str: Colourmap to use.
    - symmetric::bool: Whether to use symmetric colourbar limits.
    - title::str: Plot title.
    - axis_labels::tuple: Axis labels (x, y, z).
    - min_fraction::float: Minimum isosurface level as a fraction of maximum.
    - max_fraction::float: Maximum isosurface level as a fraction of maximum.

    # Returns:
    - fig::go.Figure: The plotly isosurface figure.
    """

    # The number of isosurface levels to plot. This can be changed, but it gets messy above 4.
    n_levels = 4

    # Filter out NaN values for min/max calculation.
    # These will appear if any range limits were applied.
    valid_data = data[~np.isnan(data)]
    data_min = valid_data.min()
    data_max = valid_data.max()

    # Flatten the coordinate arrays for the plot.
    x_flat = coord1_mesh.ravel()
    y_flat = coord2_mesh.ravel()
    z_flat = coord3_mesh.ravel()
    data_flat = data.ravel()

    fig = go.Figure()

    if symmetric:
        # Find the largest absolute value in the dataset.
        abs_max = max(abs(data_max), abs(data_min))
        max_level = abs_max * max_fraction
        min_level = abs_max * min_fraction

        # Set up shared plot parameters for both the positive and negative values.
        shared_kwargs = dict(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            value=data_flat,
            opacity=0.3,
            colorscale=colorscale,
            caps=dict(x_show=False, y_show=False, z_show=False),
            flatshading=False,
            cmin=-abs_max,
            cmax=abs_max,
            # Make it pretty.
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                specular=0.3,
                roughness=0.3,
                fresnel=0.2,
            ),
        )

        # Negative values.
        fig.add_trace(
            go.Isosurface(
                isomin=-max_level,
                isomax=-min_level,
                surface_count=n_levels,
                showscale=False,
                **shared_kwargs,
            )
        )

        # Positive values.
        fig.add_trace(
            go.Isosurface(
                isomin=min_level,
                isomax=max_level,
                surface_count=n_levels,
                showscale=True,
                colorbar=dict(
                    title=label,
                    thickness=20,
                    len=0.7,
                    x=1.05,
                ),
                **shared_kwargs,
            )
        )
    else:  # For |f_S|^2.

        # Set up the levels to plot.
        max_level = data_max * max_fraction
        min_level = data_max * min_fraction

        fig.add_trace(
            go.Isosurface(
                x=x_flat,
                y=y_flat,
                z=z_flat,
                value=data_flat,
                isomin=min_level,
                isomax=max_level,
                surface_count=n_levels,
                opacity=0.3,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=label,
                    thickness=20,
                    len=0.7,
                    x=1.05,
                ),
                caps=dict(x_show=False, y_show=False, z_show=False),
                flatshading=False,
                cmin=0.0,
                cmax=data_max,
                # Make it pretty.
                lighting=dict(
                    ambient=0.6,
                    diffuse=0.8,
                    specular=0.3,
                    roughness=0.3,
                    fresnel=0.2,
                ),
            )
        )

    # Set the axis labels and title based on the mode.
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            aspectmode="cube",
        ),
        width=1000,
        height=1000,
    )

    return fig


def main():
    # Parse the command line arguments.
    args = parse_cli_args()
    transitions_to_plot = [t.strip() for t in args.transition_indices.split(",")]

    # Resolve form factor input for one transition to avoid restarting Python per plot.
    def load_transition_data(tidx: str):
        """
        Load form factor data for a single transition. Keeps plotting in-process.

        # Arguments:
        - tidx::str: Transition index to load.

        # Returns:
        - ::tuple: Loaded data, containing e.g. form factor, transition density, and grid data.
        """
        # All three methods now use the same directory.
        tdir = Path(args.output_dir / f"transition_{tidx}")
        base = "fs_grid"
        input_path = None

        # Look for the form factor file for this transition, preferring f64 over f32.
        for candidate in [
            tdir / f"{base}_f64.h5",
            tdir / f"{base}_f32.h5",
        ]:
            if candidate.exists():
                input_path = candidate
                break

        if input_path is None:
            print(f"No {args.method} form factor file found for transition {tidx}.")
            return None

        output_dir = tdir

        output_dir.mkdir(parents=True, exist_ok=True)

        transition_density = None
        r_lim = None

        # Read the form factor data from the HDF5 file.
        with h5py.File(input_path, "r") as ff_file:
            if args.method == "spherical":
                f_s = ff_file["f_s"][()]
                theta_grid = ff_file["theta_grid"][()]
                phi_grid = ff_file["phi_grid"][()]
                q_grid = ff_file["q_grid"][()]

                # Julia writes f_s with shape (n_q, n_theta, n_phi).
                # Due to column-major/row-major differences, HDF5 reverses dimensions.
                # Python reads as (n_phi, n_theta, n_q), so transpose to (n_theta, n_phi, n_q).
                f_s = np.transpose(f_s, (1, 0, 2))

            elif args.method == "fft":
                form_factor = ff_file["form_factor"][()]
                qx_grid = ff_file["qx_grid"][()]
                qy_grid = ff_file["qy_grid"][()]
                qz_grid = ff_file["qz_grid"][()]

                # Compute q_lim from the grid.
                q_lim = np.array([qx_grid.max(), qy_grid.max(), qz_grid.max()])

                # Julia writes with shape (nx, ny, nz), but HDF5 reverses to (nz, ny, nx).
                # Transpose to get (nx, ny, nz).
                form_factor = np.transpose(form_factor, (2, 1, 0))

                if args.plot_transition_density:
                    if "transition_density" not in ff_file or "r_lim" not in ff_file:
                        print("Error: Transition density data not found in the HDF5 file.")
                        sys.exit(1)
                    transition_density = ff_file["transition_density"][()]
                    r_lim = ff_file["r_lim"][()]

                    # Julia writes with shape (nx, ny, nz), but HDF5 reverses to (nz, ny, nx).
                    # Transpose to get (nx, ny, nz).
                    transition_density = np.transpose(transition_density, (2, 1, 0))

            elif args.method == "cartesian":
                form_factor = ff_file["form_factor"][()]
                qx_grid = ff_file["qx_grid"][()]
                qy_grid = ff_file["qy_grid"][()]
                qz_grid = ff_file["qz_grid"][()]

                # Compute q_lim from the grid.
                q_lim = np.array([qx_grid.max(), qy_grid.max(), qz_grid.max()])

                # Julia writes with shape (nx, ny, nz), but HDF5 reverses to (nz, ny, nx).
                # Transpose to get (nx, ny, nz).
                form_factor = np.transpose(form_factor, (2, 1, 0))

        if args.method == "spherical":
            return (f_s, theta_grid, phi_grid, q_grid, output_dir)
        elif args.method == "fft":
            return (form_factor, q_lim, output_dir, transition_density, r_lim)
        else:
            return (form_factor, q_lim, output_dir)

    # Decide which plots to generate.
    plots_to_generate = []

    if args.method == "spherical":
        plots_to_generate.append(("form_factor", args.mode))
    elif args.method == "fft":
        plots_to_generate.append(("form_factor", args.mode))
        if args.plot_transition_density:
            plots_to_generate.append(("transition_density", None))
    elif args.method == "cartesian":
        plots_to_generate.append(("form_factor", args.mode))

    total_plots = len(plots_to_generate) * len(transitions_to_plot)
    completed = 0

    for tidx in transitions_to_plot:
        data = load_transition_data(tidx)
        if data is None:
            continue

        for plot_type, mode in plots_to_generate:
            if plot_type == "form_factor":
                # Extract the appropriate data based on the plotting mode.
                if args.method == "spherical":
                    f_s, theta_grid, phi_grid, q_grid, output_dir = data
                    plot_data, label, colorscale, symmetric = extract_domain_data(
                        f_s, mode, False
                    )

                    # Determine q_lim from args if specified.
                    q_lim_override = None
                    if (
                        args.qx_max is not None
                        or args.qy_max is not None
                        or args.qz_max is not None
                    ):
                        qx_max = args.qx_max if args.qx_max is not None else q_grid.max()
                        qy_max = args.qy_max if args.qy_max is not None else q_grid.max()
                        qz_max = args.qz_max if args.qz_max is not None else q_grid.max()
                        q_lim_override = (qx_max, qy_max, qz_max)

                    # Put the data on a uniform Cartesian grid.
                    coord1, coord2, coord3, plot_data = interpolate_to_cartesian_grid(
                        q_grid, theta_grid, phi_grid, plot_data, args.grid_size, q_lim_override
                    )

                else:
                    form_factor, q_lim, output_dir = data[0], data[1], data[2]
                    plot_data, label, colorscale, symmetric = extract_domain_data(
                        form_factor, mode, False
                    )

                    # Create the coordinate arrays.
                    nx, ny, nz = form_factor.shape
                    qx_coords = np.linspace(-q_lim[0], q_lim[0], nx)
                    qy_coords = np.linspace(-q_lim[1], q_lim[1], ny)
                    qz_coords = np.linspace(-q_lim[2], q_lim[2], nz)

                    # Downsample to target grid size.
                    coord1, coord2, coord3, plot_data = downsample_fft_grid(
                        qx_coords, qy_coords, qz_coords, plot_data, args.grid_size
                    )

                coord_type = "q"
                axis_labels = ("q<sub>x</sub> [keV]", "q<sub>y</sub> [keV]", "q<sub>z</sub> [keV]")

                # Create the plot title.
                if mode == "modsq":
                    title = f"|f<sub>S</sub>(q)|² for {args.run_name} molecule {args.molecule_number}."
                elif mode == "Im":
                    title = f"Im[f<sub>S</sub>(q)] for {args.run_name} molecule {args.molecule_number}."
                elif mode == "Re":
                    title = f"Re[f<sub>S</sub>(q)] for {args.run_name} molecule {args.molecule_number}."

                output_path = output_dir / f"form_factor_3d_{mode}.html"

            else:
                # Transition density plot (FFT method only).
                form_factor, q_lim, output_dir, transition_density, r_lim = data
                plot_data, label, colorscale, symmetric = extract_domain_data(
                    transition_density, mode, True
                )

                # Create the coordinate arrays.
                nx, ny, nz = transition_density.shape
                x_coords = np.linspace(-r_lim[0], r_lim[0], nx)
                y_coords = np.linspace(-r_lim[1], r_lim[1], ny)
                z_coords = np.linspace(-r_lim[2], r_lim[2], nz)

                # Downsample to target grid size.
                coord1, coord2, coord3, plot_data = downsample_fft_grid(
                    x_coords, y_coords, z_coords, plot_data, args.grid_size
                )

                coord_type = "r"
                axis_labels = ("x [Å]", "y [Å]", "z [Å]")
                title = f"Transition density ρ(r) for {args.run_name} molecule {args.molecule_number}."
                output_path = output_dir / "transition_density_3d.html"

            # Apply range limits.
            coord1, coord2, coord3, plot_data = apply_range_limits(
                coord1, coord2, coord3, plot_data, args, coord_type
            )

            # Create the isosurface plot.
            fig = create_isosurface_plot(
                coord1,
                coord2,
                coord3,
                plot_data,
                label,
                colorscale,
                symmetric,
                title,
                axis_labels,
                args.min_fraction,
                args.max_fraction,
            )

            # Save the plot.
            fig.write_html(str(output_path))

            completed += 1
            print(f"  Plotting {completed}/{total_plots}...", end="\r", flush=True)

    if total_plots > 0:
        print(f"Finished plotting {total_plots} 3D figure(s).")


if __name__ == "__main__":
    main()
