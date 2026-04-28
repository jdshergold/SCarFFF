# This script plots form factor slices in x-y, x-z, and y-z planes, as well as the transition density for the FFT method.
# Works with both spherical and FFT form factor outputs.

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Suppress monotonic grid warnings from pcolormesh when plotting warped grids.
# These are really annoying.
warnings.filterwarnings(
    "ignore",
    message="The input coordinates to pcolormesh are interpreted as cell centers",
    category=UserWarning,
)


def parse_cli_args():
    """
    Parse the command line arguments and return the parsed arguments.

    # Arguments:
    - None.

    # Returns:
    - parsed_args::argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot form factor slices in the x-y, x-z, and y-z planes."
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
        "--planes",
        type=str,
        nargs="+",
        default=["xy"],
        choices=["xy", "xz", "yz"],
        help="Planes to plot. The options are xy (q_z=0), xz (q_y=0), and yz (q_x=0). Default: xy.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["modsq"],
        choices=["modsq", "Im", "Re"],
        help="What to plot. The options are modsq (|f_s|^2), Im (Im(f_s)), Re (Re (f_s)). Default: modsq.",
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
        "--plot-flm-modes",
        action="store_true",
        help="Whether to plot dominant f^2_{lm}(q) modes (spherical method only).",
    )
    parser.add_argument(
        "--plot-rates",
        action="store_true",
        help="Whether to plot DM scattering rates (spherical method only).",
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

    # Parse the range arguments.
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

    if parsed_args.method not in ["spherical", "fft", "cartesian"]:
        print(f"Error: Invalid method '{parsed_args.method}'. Must be 'spherical', 'fft', or 'cartesian'.")
        sys.exit(1)

    # Set the output directory.
    parsed_args.output = runs_dir

    return parsed_args


def extract_plane_data_spherical(f_s, theta_grid, phi_grid, q_grid, plane):
    """
    Extract a 2D slice from the 3D spherical form factor data.

    # Arguments:
    - f_s::np.ndarray: The 3D array of form factor values.
    - theta_grid::np.ndarray: The theta grid.
    - phi_grid::np.ndarray: The phi grid.
    - q_grid::np.ndarray: The |q| grid.
    - plane::str: Which plane to plot. One of "xy", "xz", or "yz".

    # Returns:
    - f_s_slice::np.ndarray: A 2D array of form factor values in the specified plane.
    - coord1::np.ndarray: A 2D meshgrid of first coordinate.
    - coord2::np.ndarray: A 2D meshgrid of second coordinate.
    - label1::str: The axis label for first coordinate.
    - label2::str: The axis label for second coordinate.
    """
    # Handle the periodic boundary in phi by removing the last point if it is exactly 2pi.
    phi = phi_grid.astype(float)
    if phi.size > 1 and np.isclose(phi[-1] - phi[0], 2 * np.pi):
        phi = phi[:-1]
        f_s_trimmed = f_s[:, :-1, :]
    else:
        f_s_trimmed = f_s

    if plane == "xy":
        # In the q_z = 0 plane, theta = pi/2.
        theta_idx = int(np.abs(theta_grid - np.pi / 2).argmin())
        plane_data = f_s_trimmed[theta_idx, :, :]

        # Convert to Cartesian coordinates.
        q_mesh, phi_mesh = np.meshgrid(q_grid, phi, indexing="xy")
        coord1 = q_mesh * np.cos(phi_mesh)  # q_x = q * cos(phi).
        coord2 = q_mesh * np.sin(phi_mesh)  # q_y = q * sin(phi).
        label1 = r"$q_x\ \mathrm{[keV]}$"
        label2 = r"$q_y\ \mathrm{[keV]}$"

    elif plane == "xz":
        # In the q_y = 0 plane, we need both phi = 0 (corresponding to x > 0) and phi = pi (corresponding to x < 0).
        phi0_idx = int(np.abs(phi - 0.0).argmin())
        phiPi_idx = int(np.abs(phi - np.pi).argmin())

        plane_phi0 = f_s_trimmed[:, phi0_idx, :]
        plane_phiPi = f_s_trimmed[:, phiPi_idx, :]

        # Convert to Cartesian coordinates.
        q_mesh, theta_mesh = np.meshgrid(q_grid, theta_grid, indexing="xy")

        # In the right half, x = q*sin(theta), z = q*cos(theta).
        x0 = q_mesh * np.sin(theta_mesh)
        z0 = q_mesh * np.cos(theta_mesh)

        # In the left half, x = -q*sin(theta), z = q*cos(theta).
        xPi = -q_mesh * np.sin(theta_mesh)
        zPi = z0

        # Concatenate the two halves.
        # We need to reverse the left half, since theta = 0 corresponds to x = 0.
        coord1 = np.concatenate([xPi[:, ::-1], x0], axis=1)
        coord2 = np.concatenate([zPi[:, ::-1], z0], axis=1)
        plane_data = np.concatenate([plane_phiPi[:, ::-1], plane_phi0], axis=1)

        label1 = r"$q_x\ \mathrm{[keV]}$"
        label2 = r"$q_z\ \mathrm{[keV]}$"

    elif plane == "yz":
        # In the q_x = 0 plane, we need both phi = pi/2 (corresponding to y > 0) and phi = 3pi/2 (corresponding to y < 0).
        phiPi2_idx = int(np.abs(phi - np.pi / 2).argmin())
        phi3Pi2_idx = int(np.abs(phi - 3 * np.pi / 2).argmin())

        plane_phiPi2 = f_s_trimmed[:, phiPi2_idx, :]
        plane_phi3Pi2 = f_s_trimmed[:, phi3Pi2_idx, :]

        # Convert to Cartesian coordinates.
        q_mesh, theta_mesh = np.meshgrid(q_grid, theta_grid, indexing="xy")

        # In the upper half, y = q*sin(theta), z = q*cos(theta).
        yPi2 = q_mesh * np.sin(theta_mesh)
        zPi2 = q_mesh * np.cos(theta_mesh)

        # In the lower half, y = -q*sin(theta), z = q*cos(theta).
        y3Pi2 = -q_mesh * np.sin(theta_mesh)
        z3Pi2 = zPi2

        # Concatenate the two halves.
        # We need to reverse the lower half, since theta = 0 corresponds to y = 0.
        coord1 = np.concatenate([y3Pi2[:, ::-1], yPi2], axis=1)
        coord2 = np.concatenate([z3Pi2[:, ::-1], zPi2], axis=1)
        plane_data = np.concatenate([plane_phi3Pi2[:, ::-1], plane_phiPi2], axis=1)

        label1 = r"$q_y\ \mathrm{[keV]}$"
        label2 = r"$q_z\ \mathrm{[keV]}$"

    else:
        # Raise an error if anything other than xy, xz, or yz is provided.
        raise ValueError(
            f"Invalid plane {plane} specified. Must be one of 'xy', 'xz', or 'yz'."
        )

    return plane_data, coord1, coord2, label1, label2


def extract_plane_data_cartesian(data_3d, coord_lim, plane, coord_type="q"):
    """
    Extract a 2D slice from 3D Cartesian grid data, which can be either the form factor or transition density.

    # Arguments:
    - data_3d::np.ndarray: The 3D array of values on Cartesian grid.
    - coord_lim::np.ndarray: The coordinate limits [x_max, y_max, z_max].
    - plane::str: Which plane to plot. One of "xy", "xz", or "yz".
    - coord_type::str: Type of coordinates, this is "q" for momentum space, or "r" for real space.

    # Returns:
    - plane_data::np.ndarray: A 2D array of values in the specified plane.
    - coord1::np.ndarray: A 2D meshgrid of first coordinate.
    - coord2::np.ndarray: A 2D meshgrid of second coordinate.
    - label1::str: The axis label for first coordinate.
    - label2::str: The axis label for second coordinate.
    """
    # The data has shape (nx, ny, nz) with coordinates from -coord_lim to +coord_lim.
    nx, ny, nz = data_3d.shape
    
    # Create the coordinate arrays.
    x_coords = np.linspace(-coord_lim[0], coord_lim[0], nx)
    y_coords = np.linspace(-coord_lim[1], coord_lim[1], ny)
    z_coords = np.linspace(-coord_lim[2], coord_lim[2], nz)
    
    # Set up the axis labels based on coordinate type.
    if coord_type == "q":
        x_label = r"$q_x\ \mathrm{[keV]}$"
        y_label = r"$q_y\ \mathrm{[keV]}$"
        z_label = r"$q_z\ \mathrm{[keV]}$"
    else:
        x_label = r"$x\ \mathrm{[\AA]}$"
        y_label = r"$y\ \mathrm{[\AA]}$"
        z_label = r"$z\ \mathrm{[\AA]}$"
    
    if plane == "xy":
        # Extract the slice at z = 0.
        z_idx = nz // 2
        plane_data = data_3d[:, :, z_idx]
        
        # Create the meshgrids for plotting.
        coord1, coord2 = np.meshgrid(x_coords, y_coords, indexing="ij")
        label1 = x_label
        label2 = y_label
        
    elif plane == "xz":
        # Extract the slice at y = 0.
        y_idx = ny // 2
        plane_data = data_3d[:, y_idx, :]
        
        # Create the meshgrids for plotting.
        coord1, coord2 = np.meshgrid(x_coords, z_coords, indexing="ij")
        label1 = x_label
        label2 = z_label
        
    elif plane == "yz":
        # Extract the slice at x = 0.
        x_idx = nx // 2
        plane_data = data_3d[x_idx, :, :]
        
        # Create the meshgrids for plotting.
        coord1, coord2 = np.meshgrid(y_coords, z_coords, indexing="ij")
        label1 = y_label
        label2 = z_label
        
    else:
        raise ValueError(
            f"Invalid plane {plane} specified. Must be one of 'xy', 'xz', or 'yz'."
        )
    
    return plane_data, coord1, coord2, label1, label2


def extract_domain_data(data, mode):
    """
    Get the correct data to plot based on the selected mode.

    # Arguments:
    - data::np.ndarray: Array of form factor or transition density values.
    - mode::str: What to plot for form factors. Should be one of "modsq", "Im", or "Re". This is ignored for transition density.

    # Returns:
    - plot_data::np.ndarray: The data to plot.
    - label::str: The corresponding colourbar label.
    - cmap::str: The colourmap to use.
    - symmetric::bool: Whether to use symmetric colourbar limits.
    """
    # Check if the data is complex (form factor) or real (transition density).
    is_complex = np.iscomplexobj(data)
    
    if not is_complex:
        # We just plot the transition density directly.
        plot_data = data
        label = r"$\rho(\mathbf{r})\ [\mathrm{\AA^{-3}}]$"
        cmap = "RdBu_r"
        symmetric = True
    else:
        # The form factor is complex-valued and dimensionless.
        if mode == "modsq":
            plot_data = np.abs(data) ** 2
            label = r"$|f_s(\mathbf{q})|^2$"
            cmap = "viridis"
            symmetric = False
        elif mode == "Im":
            plot_data = np.imag(data)
            label = r"$\mathrm{Im}[f_s(\mathbf{q})]$"
            cmap = "RdBu_r"
            symmetric = True
        elif mode == "Re":
            plot_data = np.real(data)
            label = r"$\mathrm{Re}[f_s(\mathbf{q})]$"
            cmap = "RdBu_r"
            symmetric = True
        else:
            raise ValueError(
                f"Invalid mode {mode} selected. Must be one of 'modsq', 'Im', or 'Re'."
            )

    return plot_data, label, cmap, symmetric


def apply_plot_limits(coord1, coord2, plane_data, args, plane, data_is_complex=None):
    """
    Apply the plot range limits specified in args.
    
    # Arguments:
    - coord1::np.ndarray: First coordinate meshgrid.
    - coord2::np.ndarray: Second coordinate meshgrid.
    - plane_data::np.ndarray: Data to plot.
    - args::argparse.Namespace: Command line arguments.
    - plane::str: Which plane is being plotted.
    - data_is_complex::bool or None: Whether the underlying data is complex-valued.
      If None, this is inferred from plane_data.
    
    # Returns:
    - coord1_limited::np.ndarray: Limited first coordinate meshgrid.
    - coord2_limited::np.ndarray: Limited second coordinate meshgrid.
    - plane_data_limited::np.ndarray: Limited data.
    - xlim::tuple: x-axis limits for plotting.
    - ylim::tuple: y-axis limits for plotting.
    """
    # Check if the data is complex (form factor) or real (transition density).
    if data_is_complex is None:
        is_complex = np.iscomplexobj(plane_data)
    else:
        is_complex = data_is_complex
    
    # Determine which coordinates correspond to which axes for this plane.
    if not is_complex:
        # Use real-space coordinates (x, y, z) for the transition density.
        if plane == "xy":
            coord1_min, coord1_max = args.x_min, args.x_max
            coord2_min, coord2_max = args.y_min, args.y_max
        elif plane == "xz":
            coord1_min, coord1_max = args.x_min, args.x_max
            coord2_min, coord2_max = args.z_min, args.z_max
        elif plane == "yz":
            coord1_min, coord1_max = args.y_min, args.y_max
            coord2_min, coord2_max = args.z_min, args.z_max
    else:
        # Use momentum-space coordinates (qx, qy, qz) for the form factor.
        if plane == "xy":
            coord1_min, coord1_max = args.qx_min, args.qx_max
            coord2_min, coord2_max = args.qy_min, args.qy_max
        elif plane == "xz":
            coord1_min, coord1_max = args.qx_min, args.qx_max
            coord2_min, coord2_max = args.qz_min, args.qz_max
        elif plane == "yz":
            coord1_min, coord1_max = args.qy_min, args.qy_max
            coord2_min, coord2_max = args.qz_min, args.qz_max
    
    # For Cartesian grids (FFT and Cartesian methods), we can slice the data.
    # Check if this is a regular Cartesian grid.
    is_cartesian = (coord1.ndim == 2 and coord1.shape[0] > 1 and coord1.shape[1] > 1 and
                    np.allclose(coord1[:, 0], coord1[:, -1]) and
                    np.allclose(coord2[0, :], coord2[-1, :]))

    if is_cartesian:
        # Extract 1D coordinate arrays.
        coord1_1d = coord1[:, 0]
        coord2_1d = coord2[0, :]

        # Find the indices for slicing based on the specified ranges.
        if coord1_min is not None:
            i_start = np.searchsorted(coord1_1d, coord1_min)
        else:
            i_start = 0

        if coord1_max is not None:
            i_end = np.searchsorted(coord1_1d, coord1_max, side='right')
        else:
            i_end = len(coord1_1d)

        if coord2_min is not None:
            j_start = np.searchsorted(coord2_1d, coord2_min)
        else:
            j_start = 0

        if coord2_max is not None:
            j_end = np.searchsorted(coord2_1d, coord2_max, side='right')
        else:
            j_end = len(coord2_1d)

        # Slice the data.
        coord1 = coord1[i_start:i_end, j_start:j_end]
        coord2 = coord2[i_start:i_end, j_start:j_end]
        plane_data = plane_data[i_start:i_end, j_start:j_end]

        # Set the limits to the actual sliced data range.
        xlim = (coord1.min(), coord1.max())
        ylim = (coord2.min(), coord2.max())
    else:
        # For irregular grids (spherical), just set the axis limits without slicing.
        xlim = (coord1_min if coord1_min is not None else coord1.min(),
                coord1_max if coord1_max is not None else coord1.max())
        ylim = (coord2_min if coord2_min is not None else coord2.min(),
                coord2_max if coord2_max is not None else coord2.max())

    return coord1, coord2, plane_data, xlim, ylim



def plot_flm_modes(f_lm, q_grid, output_dir, top_n=8):
    """
    Plot the top N f^2_{lm}(q) modes by integrated absolute area on a single set of axes.
    Lines are grouped by l (same colour) with distinct linestyles for different m.

    # Arguments:
    - f_lm::np.ndarray: The f_lm tensor with shape (n_q, n_keys).
    - q_grid::np.ndarray: The |q| grid in keV.
    - output_dir::Path: Directory to save the output plot.
    - top_n::int: Number of top modes to plot, ranked by integrated absolute area (default: 8).
    """
    n_keys = f_lm.shape[1]
    l_max = int(np.sqrt(n_keys)) - 1
    areas = {}
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            key = l * l + (l + m) + 1
            areas[(l, m)] = np.trapz(np.abs(f_lm[:, key - 1]), q_grid)
    top_modes = sorted(areas, key=areas.get, reverse=True)[:top_n]

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, (l, m) in enumerate(sorted(top_modes)):
        key = l * l + (l + m) + 1
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        label = rf"$({l},\,{m})$"
        ax.plot(q_grid, f_lm[:, key - 1], lw=1.2, ls=ls, color=color, label=label)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$q\ [\mathrm{keV}]$")
    ax.set_ylabel(r"$f^2_{\ell m}(q)$")
    ax.set_xlim(q_grid[0], q_grid[-1])
    ax.legend(fontsize=8, ncol=2, title=r"Dominant modes: $(\ell,\,m)$", title_fontsize=8)
    fig.tight_layout()

    fig.savefig(output_dir / "flm_modes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scattering_rates(spherical_dir):
    """
    Plot the DM scattering rates as a function of DM mass.

    # Arguments:
    - spherical_dir::Path: Directory containing the scattering_rates HDF5 file.
    """
    input_path = None
    for candidate in [
        spherical_dir / "scattering_rates_f64.h5",
        spherical_dir / "scattering_rates_f32.h5",
    ]:
        if candidate.exists():
            input_path = candidate
            break

    if input_path is None:
        print("  Warning: scattering rates not found. Was COMPUTE_RATES=true?")
        return False

    with h5py.File(input_path, "r") as rate_file:
        mchi = rate_file["mchi_MeV"][()]
        rate_max = rate_file["rate_max"][()]
        rate_min = rate_file["rate_min"][()]
        rate_mean = rate_file["rate_mean"][()]

    fig, ax = plt.subplots(figsize=(7, 4))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.plot(mchi, rate_max, lw=1.2, ls="-",  color=colors[0], label=r"Maximum")
    ax.plot(mchi, rate_min, lw=1.2, ls="--", color=colors[1], label=r"Minimum")
    ax.plot(mchi, rate_mean, lw=1.2, ls="-.", color=colors[2], label=r"Mean")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_\chi\ [\mathrm{MeV}]$")
    ax.set_ylabel(r"Dimensionless rate")
    ax.set_xlim(mchi[0], mchi[-1])
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()

    fig.savefig(spherical_dir / "scattering_rates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    # Parse the command line arguments.
    args = parse_cli_args()
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    runs_dir = project_root / "runs" / args.run_name / str(args.molecule_number)
    transitions_to_plot = [t.strip() for t in args.transition_indices.split(",")]

    # Check if molecule directory exists.
    if not runs_dir.exists():
        print(f"Error: Molecule directory not found at {runs_dir}")
        print(
            f"Please run TD-DFT for run '{args.run_name}' first."
        )
        sys.exit(1)

    # Set plotting parameters.
    mpl.rcParams.update(
        {
            "font.family":      "serif",
            "font.serif":       ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size":        11,
        }
    )

    # Helper to plot one transition without restarting Python.
    def plot_transition(tidx: str):
        """
        Plot slices for a single transition. Keeps plotting in-process to avoid repeated Python start-up.

        # Arguments:
        - tidx::str: Transition index to plot.
        """
        tdir = runs_dir / args.method / f"transition_{tidx}"
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
            return []
        output_dir = tdir

        output_dir.mkdir(parents=True, exist_ok=True)

        # Read the form factor data from HDF5 file.
        with h5py.File(input_path, "r") as ff_file:
            if args.method == "spherical":
                q_grid = ff_file["q_grid"][()]

                f_s = None
                theta_grid = None
                phi_grid = None
                if "f_s" in ff_file:
                    f_s = ff_file["f_s"][()]
                    theta_grid = ff_file["theta_grid"][()]
                    phi_grid = ff_file["phi_grid"][()]

                    # Julia writes f_s with shape (n_q, n_theta, n_phi).
                    # Due to column-major/row-major differences, HDF5 reverses dimensions.
                    # Python reads as (n_phi, n_theta, n_q), so transpose to (n_theta, n_phi, n_q).
                    f_s = np.transpose(f_s, (1, 0, 2))

                # Load f_lm if requested, and transpose because of the usual Julia vs Python nonsense.
                f_lm = None
                if args.plot_flm_modes and "f_lm" in ff_file:
                    f_lm = ff_file["f_lm"][()].T
                elif args.plot_flm_modes:
                    print(f"  Warning: f_lm not found for transition {tidx}. Was f_lm_tensor included in COMPUTE_MODES?")
                
            elif args.method == "fft":
                # Load form factor data.
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
                    # Also load the transition density data.
                    transition_density = ff_file["transition_density"][()]
                    r_lim = ff_file["r_lim"][()]

                    # Julia writes with shape (nx, ny, nz), but HDF5 reverses to (nz, ny, nx).
                    # Transpose to get (nx, ny, nz).
                    transition_density = np.transpose(transition_density, (2, 1, 0))

            elif args.method == "cartesian":
                # Load form factor data.
                form_factor = ff_file["form_factor"][()]
                qx_grid = ff_file["qx_grid"][()]
                qy_grid = ff_file["qy_grid"][()]
                qz_grid = ff_file["qz_grid"][()]

                # Compute q_lim from the grid.
                q_lim = np.array([qx_grid.max(), qy_grid.max(), qz_grid.max()])

                # Julia writes with shape (nx, ny, nz), but HDF5 reverses to (nz, ny, nx).
                # Transpose to get (nx, ny, nz).
                form_factor = np.transpose(form_factor, (2, 1, 0))

        # Return data needed for plotting without re-reading from disk.
        if args.method == "spherical":
            return [f_s, theta_grid, phi_grid, q_grid, output_dir, f_lm if args.plot_flm_modes else None]
        elif args.method == "fft":
            return [None, None, None, None, form_factor, q_lim, output_dir, transition_density if args.plot_transition_density else None, r_lim if args.plot_transition_density else None]
        else:
            return [None, None, None, None, form_factor, q_lim, output_dir]

    # Decide which plots to generate.
    plots_to_generate = []

    if args.method == "spherical":
        plots_to_generate.append(("form_factor", False))
        if args.plot_flm_modes:
            plots_to_generate.append(("flm_modes", False))
    elif args.method == "fft":
        plots_to_generate.append(("form_factor", False))
        if args.plot_transition_density:
            plots_to_generate.append(("transition_density", True))
    elif args.method == "cartesian":
        plots_to_generate.append(("form_factor", False))
    
    plot_rates = args.method == "spherical" and args.plot_rates
    total_plots = len(plots_to_generate) * len(transitions_to_plot) + int(plot_rates)
    completed = 0

    if plot_rates:
        if plot_scattering_rates(runs_dir / args.method):
            completed += 1
            print(f"  Plotting {completed}/{total_plots}...", end="\r", flush=True)
        else:
            total_plots -= 1

    for tidx in transitions_to_plot:
        run_outputs = plot_transition(tidx)
        if not run_outputs:
            continue
        output_dir = run_outputs[4] if args.method == "spherical" else run_outputs[6]

        for plot_type, is_transition_density in plots_to_generate:
            # Skip form factor plot if f_s wasn't computed.
            if plot_type == "form_factor" and args.method == "spherical" and run_outputs[0] is None:
                completed += 1
                continue

            # Determine which modes to plot.
            if is_transition_density:
                # We only plot the transition density, no modes.
                modes_to_plot = [None]  # Single plot, no mode label.

            # Plot the f_lm^2 coefficients.
            elif plot_type == "flm_modes":
                if run_outputs[5] is not None:
                    plot_flm_modes(run_outputs[5], run_outputs[3], output_dir)
                completed += 1
                print(f"  Plotting {completed}/{total_plots}...", end="\r", flush=True)
                continue

            else:
                # For the form factor, plot the selected modes.
                modes_to_plot = args.modes
            
            # Determine the subplot grid size.
            n_modes = len(modes_to_plot)
            n_planes = len(args.planes)

            # Create figure with subplots arranged as modes * planes.
            # Each mode gets a row, each plane gets a column.
            fig_width = 5.5 * n_planes
            fig_height = 5.0 * n_modes
            fig, axes = plt.subplots(
                n_modes, n_planes, figsize=(fig_width, fig_height), squeeze=False
            )

            # Iterate over modes (rows) and planes (columns).
            for mode_idx, mode in enumerate(modes_to_plot):
                for plane_idx, plane in enumerate(args.planes):
                    ax = axes[mode_idx, plane_idx]

                    # Extract the plane data based on method and data type.
                    if args.method == "spherical":
                        plane_data, coord1, coord2, label1, label2 = extract_plane_data_spherical(
                            run_outputs[0], run_outputs[1], run_outputs[2], run_outputs[3], plane
                        )
                    elif args.method == "fft":
                        if is_transition_density:
                            plane_data, coord1, coord2, label1, label2 = extract_plane_data_cartesian(
                                run_outputs[7], run_outputs[8], plane, coord_type="r"
                            )
                        else:
                            plane_data, coord1, coord2, label1, label2 = extract_plane_data_cartesian(
                                run_outputs[4], run_outputs[5], plane, coord_type="q"
                            )
                    elif args.method == "cartesian":
                        plane_data, coord1, coord2, label1, label2 = extract_plane_data_cartesian(
                            run_outputs[4], run_outputs[5], plane, coord_type="q"
                        )

                    # Extract the domain data.
                    domain_is_complex = np.iscomplexobj(plane_data)
                    plot_data, cbar_label, cmap, symmetric = extract_domain_data(
                        plane_data, mode
                    )

                    # Apply the plot limits.
                    coord1, coord2, plot_data, xlim, ylim = apply_plot_limits(
                        coord1,
                        coord2,
                        plot_data,
                        args,
                        plane,
                        data_is_complex=domain_is_complex,
                    )

                    # Set the colourbar limits.
                    if symmetric:
                        vmax = np.abs(plot_data).max()
                        vmin = -vmax
                    else:
                        vmin = 0
                        vmax = plot_data.max()
                        
                    # Create the plot using pcolormesh, which handles irregular grids correctly.
                    pcm = ax.pcolormesh(
                        coord1,
                        coord2,
                        plot_data,
                        shading="auto",
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )

                    # Set the axis labels and limits.
                    ax.set_xlabel(label1)
                    ax.set_ylabel(label2)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_aspect("equal", adjustable="box")

                    # Add the colourbar for each subplot.
                    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
                    cbar.set_label(cbar_label, rotation=270, labelpad=20)

            # Save the plot.
            fig.tight_layout()
            
            # Generate the output filename for this plot type.
            planes_str = "_".join(args.planes)
            if is_transition_density:
                # For the transition density, we don't have to specify a mode.
                output_path = output_dir / f"transition_density_{planes_str}.png"
            else:
                # For the form factor, include the modes in the filename.
                modes_str = "_".join(modes_to_plot)
                output_path = output_dir / f"form_factor_{planes_str}_{modes_str}.png"
            
            fig.savefig(output_path, dpi=300, bbox_inches="tight")

            # Close the plot.
            plt.close(fig)

            completed += 1
            print(f"  Plotting {completed}/{total_plots}...", end="\r", flush=True)

    if total_plots > 0:
        print(f"Finished plotting {total_plots} figure(s).")


if __name__ == "__main__":
    main()
