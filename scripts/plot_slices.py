# This script plots form factor slices in x-y, x-z, and y-z planes, as well as the transition density for the FFT method.
# Works with both spherical and FFT form factor outputs.

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import warnings

from plot_quantities import crystal_grid_key, crystal_group_grid, crystal_quantity_label, read_structure_function


# Suppress monotonic grid warnings from pcolormesh when plotting warped grids.
# These are really annoying.
warnings.filterwarnings(
    "ignore",
    message="The input coordinates to pcolormesh are interpreted as cell centers",
    category=UserWarning,
)

# Crystal state grouping. We group by n_mols * n_degenerate, so e.g. a 4 molecule cell with
# 3 degenerate states per monomer would e.g. have 12 states grouped into one for plotting.

def cluster_monomer_transitions(monomer_energies, threshold):
    """
    Group monomer transitions whose energies are within a threshold of their neighbour.

    # Arguments:
    - monomer_energies::np.ndarray: The monomer transition energies in eV, ascending.
    - threshold::float: Energies closer than this to the previous one join its cluster, in eV.

    # Returns:
    - list[list[int]]: The transition indices in each cluster, in ascending energy.
    """

    ordered = np.sort(np.asarray(monomer_energies))
    clusters = [[0]]
    for transition in range(1, len(ordered)):
        if ordered[transition] - ordered[transition - 1] <= threshold:
            clusters[-1].append(transition)
        else:
            clusters.append([transition])

    return clusters


def build_state_groups(monomer_energies, n_states_per_transition, threshold, group_size=0):
    """
    Work out which crystal states belong in each plotting group.

    # Arguments:
    - monomer_energies::np.ndarray: The monomer transition energies in eV.
    - n_states_per_transition::int: The number of stored states per monomer transition. In the coherent case this is
      the number of molecular images, which is how many bands each monomer transition gives rise to.
    - threshold::float: Degeneracy threshold in eV, used when group_size is not given.
    - group_size::int: A fixed number of states per group, overriding the clustering. Zero uses the
      clustering, and one gives a group per state.

    # Returns:
    - list[tuple[int, int]]: The half open state index range of each group, zero based.
    - list[int]: How many monomer transitions each group covers.
    """

    n_states = n_states_per_transition * len(monomer_energies)

    if group_size > 0:
        if n_states % group_size:
            raise ValueError(f"{n_states} states do not divide into groups of {group_size}.")
        bounds = [(start, start + group_size) for start in range(0, n_states, group_size)]
        return bounds, [group_size / n_states_per_transition] * len(bounds)

    bounds = []
    degeneracies = []
    start = 0
    for cluster in cluster_monomer_transitions(monomer_energies, threshold):
        size = n_states_per_transition * len(cluster)
        bounds.append((start, start + size))
        degeneracies.append(len(cluster))
        start += size

    return bounds, degeneracies


def describe_groups(bounds, degeneracies, n_states_per_transition, band_min, band_max, band_mean):
    """
    Build a human readable summary of the grouping, and say which boundaries can still be crossed.

    A group boundary whose neighbouring energy ranges overlap can still have a band cross it, which
    leaves a discontinuity that no fixed grouping removes. It is worth naming rather than leaving to
    look like noise.

    # Arguments:
    - bounds::list[tuple[int, int]]: The state index range of each group.
    - degeneracies::list[int]: Monomer transitions covered by each group.
    - n_states_per_transition::int: Molecules per unit cell.
    - band_min::np.ndarray: Per state minimum band energy in eV.
    - band_max::np.ndarray: Per state maximum band energy in eV.
    - band_mean::np.ndarray: Per state mean band energy in eV.

    # Returns:
    - list[str]: Lines describing the grouping.
    """

    lines = [f"{len(bounds)} groups from {n_states_per_transition} molecules per cell:"]
    lows = [band_min[first:last].min() for first, last in bounds]
    highs = [band_max[first:last].max() for first, last in bounds]

    for index, ((first, last), degeneracy) in enumerate(zip(bounds, degeneracies)):
        lines.append(
            f"  group {index + 1:2d}: states {first + 1:3d}-{last:3d} "
            f"({last - first:2d} states, {degeneracy:g}-fold monomer degeneracy), "
            f"mean {band_mean[first:last].mean():.4f} eV, "
            f"range {lows[index]:.4f}-{highs[index]:.4f} eV")

    overlapping = [i for i in range(len(bounds) - 1) if lows[i + 1] < highs[i]]
    if overlapping:
        pairs = ", ".join(f"{i + 1}/{i + 2}" for i in overlapping)
        lines.append(f"  boundaries whose energy ranges still overlap: {pairs}. Bands can cross "
                     f"these, so a little discontinuity survives there.")
    else:
        lines.append("  no group boundaries overlap in energy, so no bands cross between groups.")

    return lines



def plot_structure_factor(crystal_path, output_dir, top_n=8, dpi=200):
    """
    Draw S_{ell m}(q,E), including 2π/V_cell, over the (q,E) plane, one panel per mode.

    # Arguments:
    - crystal_path::Path: The crystal HDF5, carrying S_lm and E_grid_eV.
    - output_dir::Path: Directory to write the figure into.
    - top_n::int: How many of the dominant (l, m) modes to draw beside the monopole.
    - dpi::int: Figure resolution.

    # Returns:
    - None.
    """

    with h5py.File(crystal_path, "r") as crystal_file:
        # Julia writes (n_energy, n_q, n_keys), and HDF5 reverses it.
        structure_factor, energy_grid, q_grid = read_structure_function(crystal_file)

    # Compute the step size.
    step = (energy_grid[-1] - energy_grid[0]) / (len(energy_grid) - 1)

    # Use the same top modes as the energy-independent form factor, since their integrated weight should be the same.
    integrated = structure_factor.sum(axis=0) * step
    modes = [mode for mode in get_top_f_sq_lm_modes(integrated, q_grid, top_n=top_n + 1)
             if mode != (0, 0)][:top_n]

    # Always plot the isotropic mode first.
    panels = [(0, 0)] + modes
    n_columns = min(3, len(panels))
    n_rows = int(np.ceil(len(panels) / n_columns))

    figure, axes = plt.subplots(n_rows, n_columns, figsize=(4.6 * n_columns, 3.6 * n_rows),
                                squeeze=False, sharex=True, sharey=True, layout="constrained")

    # Use the same scale for all modes, to makew them easier to compare.
    limit = max(np.abs(structure_factor[:, :, l * l + (l + m)]).max() for l, m in panels) or 1.0
    norm = SymLogNorm(linthresh=limit * 1e-3, vmin=-limit, vmax=limit, base=10)

    image = None
    for index, (l, m) in enumerate(panels):
        row, column = divmod(index, n_columns)
        axis = axes[row][column]
        image = axis.pcolormesh(q_grid, energy_grid, structure_factor[:, :, l * l + (l + m)],
                                shading="nearest", cmap="RdBu_r", norm=norm)
        axis.set_title(rf"$\ell = {l},\ m = {m}$")
        if row == n_rows - 1:
            axis.set_xlabel(r"$q$ (keV)")
        if column == 0:
            axis.set_ylabel(r"$E$ (eV)")

    # Any panels left over on the last row would otherwise render as empty axes.
    for index in range(len(panels), n_rows * n_columns):
        row, column = divmod(index, n_columns)
        axes[row][column].axis("off")

    figure.colorbar(image, ax=axes, shrink=0.6,
                    label=r"$S_{\ell m}(q,E)$ [Å$^{-3}$ eV$^{-1}$]")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "structure_factor_qE.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_band_energies(crystal_path, output_dir, dpi=200):
    """
    Draw ΔE_n(k) over the sampled Brillouin zone planes, one figure per crystal state.
    Points outside the zone are NaN and render blank, which will happen for some space groups.

    # Arguments:
    - crystal_path::Path: The coherent crystal HDF5, carrying the maps in its band_map group.
    - output_dir::Path: Directory to write the per state figures into.
    - dpi::int: Figure resolution.

    # Returns:
    - None.
    """

    labels = {"xy": ("k_x", "k_y"), "xz": ("k_x", "k_z"), "yz": ("k_y", "k_z")}

    with h5py.File(crystal_path, "r") as crystal_file:
        band_group = crystal_file["band_map"]
        planes = [p.decode() if isinstance(p, bytes) else p for p in band_group["planes"][()]]
        maps = {}
        for plane in planes:
            # Julia writes (n_states, n_a, n_b); HDF5 reverses it, so transpose back.
            maps[plane] = (np.array(band_group[plane]["Delta_E_eV"]).T,
                           band_group[plane]["axis_a_keV"][()],
                           band_group[plane]["axis_b_keV"][()])

    n_states = maps[planes[0]][0].shape[0]
    n_planes = len(planes)
    output_dir.mkdir(parents=True, exist_ok=True)

    for state in range(n_states):
        figure, axes = plt.subplots(1, n_planes, figsize=(5.5 * n_planes, 5.0), squeeze=False)
        for plane_index, plane in enumerate(planes):
            energies, axis_a, axis_b = maps[plane]
            axis = axes[0, plane_index]
            image = axis.pcolormesh(axis_a, axis_b, energies[state].T,
                                    shading="nearest", cmap="viridis")
            first_label, second_label = labels[plane]
            axis.set_xlabel(rf"${first_label}$ (keV)")
            axis.set_ylabel(rf"${second_label}$ (keV)")
            figure.colorbar(image, ax=axis, label=r"$\Delta E_n(\mathbf{k})\,[\mathrm{eV}]$")

        figure.suptitle(rf"Frenkel exciton band $n={state + 1}$ over the first Brillouin zone")
        figure.tight_layout()
        figure.savefig(output_dir / f"state_{state + 1:02d}.png", dpi=dpi, bbox_inches="tight")
        plt.close(figure)

    print(f"Band energies saved to {output_dir} ({n_states} states, "
          f"{n_planes} plane(s): {', '.join(planes)}).")


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
        "--results-dir",
        type=str,
        default=None,
        help="Relative results directory under runs/<run-name>/<molecule-number>/. Defaults to the method directory, or crystal/ for top-level crystal spherical plots.",
    )
    parser.add_argument(
        "--transition-indices",
        type=str,
        default=None,
        help="Transition indices to plot (spherical and Cartesian methods). Comma-separated. "
             "Defaults to 1, or to every group for an aggregate crystal output, where the point is "
             "to see all of them.",
    )
    parser.add_argument(
        "--planes",
        type=str,
        nargs="+",
        default=None,
        choices=["xy", "xz", "yz"],
        help="Planes to plot. The options are xy (q_z=0), xz (q_y=0), and yz (q_x=0). Defaults to "
             "xy, or to all three for an aggregate crystal output, where they go in one figure.",
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
        "--group-states",
        action="store_true",
        help="Group near-degenerate crystal contributions for plotting, with --transition-indices "
             "then selecting the group. Coherent groups contain crystal states, whilst incoherent groups "
             "contain monomer transitions.",
    )
    parser.add_argument(
        "--degeneracy-threshold",
        type=float,
        default=0.05,
        help="Monomer transitions closer than this in eV are treated as degenerate and their state "
             "blocks are grouped together (default: 0.05).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="Override the degeneracy clustering with a fixed number of states per group.",
    )
    parser.add_argument(
        "--plot-transition-density",
        action="store_true",
        help="Whether to plot the transition density (FFT method only).",
    )
    parser.add_argument(
        "--plot-f-sq-lm-modes",
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
    has_crystal_metadata = (runs_dir / "crystal_metadata.json").exists()
    if parsed_args.results_dir is not None:
        parsed_args.results_root = runs_dir / parsed_args.results_dir
    elif parsed_args.method == "spherical" and has_crystal_metadata:
        parsed_args.results_root = runs_dir / "crystal"
    else:
        parsed_args.results_root = runs_dir / parsed_args.method

    parsed_args.output = runs_dir
    parsed_args.crystal_mode = (
        parsed_args.method == "spherical"
        and has_crystal_metadata
        and parsed_args.results_root == runs_dir / "crystal"
    )

    # A crystal run puts everything in one file, named based on whatever method was used.
    parsed_args.crystal = False
    parsed_args.crystal_path = None
    parsed_args.crystal_order = None
    if parsed_args.crystal_mode:
        for candidate in ["crystal_form_factors_f64.h5", "crystal_form_factors_f32.h5"]:
            path = parsed_args.results_root / candidate
            if path.exists():
                parsed_args.crystal_path = path
                break

    if parsed_args.crystal_path is not None:
        with h5py.File(parsed_args.crystal_path, "r") as order_file:
            order = order_file["crystal_order"][()]
        parsed_args.crystal_order = order.decode() if isinstance(order, bytes) else str(order)
        parsed_args.crystal = True

        with h5py.File(parsed_args.crystal_path, "r") as crystal_file:
            # f_s is only written when the form factor itself was asked for, whereas state_f_sq_lm
            # always is, so take the state count from state_f_sq_lm and note whether there is a grid
            # to slice at all. Without one the bands can still be drawn.
            parsed_args.has_grid = any(key in crystal_file for key in ("f_uc", "I_uc"))
            parsed_args.original_transitions = crystal_file["transition_indices"][()]
            n_states = crystal_file["f_sq_lm"].shape[2]
            n_transitions = crystal_file["transition_indices"].shape[0]
            parsed_args.n_molecules = len(np.unique(crystal_file["image_of"][()]))
            localised = crystal_file["Delta_E_monomer_eV"][()]
            # Band statistics exist only for the coherent order, since the incoherent has no bands.
            has_bands = "Delta_E_band_mean_eV" in crystal_file
            band_min = crystal_file["Delta_E_band_min_eV"][()] if has_bands else None
            band_max = crystal_file["Delta_E_band_max_eV"][()] if has_bands else None
            band_mean = crystal_file["Delta_E_band_mean_eV"][()] if has_bands else None

        # Coherent groups run over crystal states, so each monomer transition contributes n_states_per_transition of
        # them; incoherent groups run over the monomer transitions themselves, so n_states_per_transition is 1 and the
        # same clustering gives one group per degenerate manifold either way.
        n_states_per_transition = n_states // n_transitions
        if parsed_args.group_states or parsed_args.group_size > 0:
            # The basis runs over transitions fastest, so the first block of localised energies is
            # the monomer transition energies.
            monomer_energies = localised[:n_transitions]
            try:
                parsed_args.group_bounds, degeneracies = build_state_groups(
                    monomer_energies, n_states_per_transition, parsed_args.degeneracy_threshold,
                    group_size=parsed_args.group_size)
            except ValueError as error:
                print(error)
                sys.exit(1)
            if has_bands:
                for line in describe_groups(parsed_args.group_bounds, degeneracies, n_states_per_transition,
                                            band_min, band_max, band_mean):
                    print(line)
            else:
                print(f"{parsed_args.n_groups if hasattr(parsed_args, 'n_groups') else len(parsed_args.group_bounds)} "
                      f"groups from {n_transitions} monomer transitions "
                      f"({parsed_args.crystal_order}):")
                for index, (first, last) in enumerate(parsed_args.group_bounds):
                    print(f"  group {index + 1:2d}: transitions {first + 1:3d}-{last:3d} "
                          f"({last - first:2d}), mean {np.mean(localised[first:last]):.4f} eV")
        else:
            parsed_args.group_bounds = [(s, s + 1) for s in range(n_states)]
        parsed_args.n_groups = len(parsed_args.group_bounds)

        # Aggregate crystal indices are groups, and there is no reason to want only the first, so
        # plot the lot unless the caller asked for particular ones.
        if parsed_args.transition_indices is None:
            parsed_args.transition_indices = ",".join(
                str(group) for group in range(1, parsed_args.n_groups + 1))
        if parsed_args.planes is None:
            parsed_args.planes = ["xy", "xz", "yz"]

        # Summing |f|^2 over a group discards the phase, so the signed modes have nothing to show.
        grouped = any(last - first > 1 for first, last in parsed_args.group_bounds)
        if (grouped or parsed_args.crystal_order == "incoherent") and any(m != "modsq" for m in parsed_args.modes):
            print("Crystal intensity plots average |f|^2, so only --modes modsq is available; "
                  "signed modes need a single coherent band (--group-size 1).")
            sys.exit(1)
        parsed_args.grouped = grouped

    return parsed_args


def angular_slice(data, grid, angle, axis):
    """
    Interpolate a spherical grid to the angle defining a Cartesian plane.

    # Arguments:
    - data::np.ndarray: The spherical grid values.
    - grid::np.ndarray: The sampled angles, in radians.
    - angle::float: The angle to evaluate, in radians.
    - axis::int: The angular axis in data.

    # Returns:
    - np.ndarray: The values at the requested angle.
    """
    upper = int(np.searchsorted(grid, angle))
    if upper < len(grid) and np.isclose(grid[upper], angle, rtol=0, atol=1e-12):
        return np.take(data, upper, axis=axis)
    if upper == 0 or upper == len(grid):
        raise ValueError("The requested plane is outside the angular grid.")
    lower = upper - 1
    weight = (angle - grid[lower]) / (grid[upper] - grid[lower])
    return (1 - weight) * np.take(data, lower, axis=axis) + weight * np.take(data, upper, axis=axis)


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

    # Close the periodic grid so the slices cover every azimuth, including the seam.
    phi = np.append(phi, phi[0] + 2 * np.pi)
    f_s_trimmed = np.concatenate((f_s_trimmed, f_s_trimmed[:, :1, :]), axis=1)

    if plane == "xy":
        # Interpolate to theta = pi/2 if the grid does not contain it exactly.
        plane_data = angular_slice(f_s_trimmed, theta_grid, np.pi / 2, axis=0)

        # Convert to Cartesian coordinates.
        q_mesh, phi_mesh = np.meshgrid(q_grid, phi, indexing="xy")
        coord1 = q_mesh * np.cos(phi_mesh)  # q_x = q * cos(phi).
        coord2 = q_mesh * np.sin(phi_mesh)  # q_y = q * sin(phi).
        label1 = r"$q_x\ \mathrm{[keV]}$"
        label2 = r"$q_y\ \mathrm{[keV]}$"

    elif plane == "xz":
        # In the q_y = 0 plane, we need both phi = 0 (corresponding to x > 0) and phi = pi (corresponding to x < 0).
        plane_phi0 = angular_slice(f_s_trimmed, phi, 0.0, axis=1)
        plane_phiPi = angular_slice(f_s_trimmed, phi, np.pi, axis=1)

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
        plane_phiPi2 = angular_slice(f_s_trimmed, phi, np.pi / 2, axis=1)
        plane_phi3Pi2 = angular_slice(f_s_trimmed, phi, 3 * np.pi / 2, axis=1)

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
    - coord_lim::tuple or np.ndarray: The saved momentum axes, or the FFT real-space half widths.
    - plane::str: Which plane to plot. One of "xy", "xz", or "yz".
    - coord_type::str: Type of coordinates, this is "q" for momentum space, or "r" for real space.

    # Returns:
    - plane_data::np.ndarray: A 2D array of values in the specified plane.
    - coord1::np.ndarray: A 2D meshgrid of first coordinate.
    - coord2::np.ndarray: A 2D meshgrid of second coordinate.
    - label1::str: The axis label for first coordinate.
    - label2::str: The axis label for second coordinate.
    """
    nx, ny, nz = data_3d.shape

    # Momentum axes come from the saved grids. The FFT spatial grid excludes its upper endpoint.
    if coord_type == "q":
        x_coords, y_coords, z_coords = coord_lim
    else:
        x_coords = np.linspace(-coord_lim[0], coord_lim[0], nx, endpoint=False)
        y_coords = np.linspace(-coord_lim[1], coord_lim[1], ny, endpoint=False)
        z_coords = np.linspace(-coord_lim[2], coord_lim[2], nz, endpoint=False)

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
        z_idx = int(np.argmin(np.abs(z_coords)))
        plane_data = data_3d[:, :, z_idx]
        
        # Create the meshgrids for plotting.
        coord1, coord2 = np.meshgrid(x_coords, y_coords, indexing="ij")
        label1 = x_label
        label2 = y_label
        
    elif plane == "xz":
        # Extract the slice at y = 0.
        y_idx = int(np.argmin(np.abs(y_coords)))
        plane_data = data_3d[:, y_idx, :]
        
        # Create the meshgrids for plotting.
        coord1, coord2 = np.meshgrid(x_coords, z_coords, indexing="ij")
        label1 = x_label
        label2 = z_label
        
    elif plane == "yz":
        # Extract the slice at x = 0.
        x_idx = int(np.argmin(np.abs(x_coords)))
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


def extract_domain_data(data, mode, already_squared=False, is_transition_density=False):
    """
    Get the correct data to plot based on the selected mode.

    # Arguments:
    - data::np.ndarray: Array of form factor or transition density values.
    - mode::str: What to plot for form factors. Should be one of "modsq", "Im", or "Re". This is ignored for transition density.
    - is_transition_density::bool: Whether the data is a real-space transition density.
    - already_squared::bool: Whether the data is already |f|^2, which it is when crystal states have
      been summed over a group, since the sum has to happen after squaring.

    # Returns:
    - plot_data::np.ndarray: The data to plot.
    - label::str: The corresponding colourbar label.
    - cmap::str: The colourmap to use.
    - symmetric::bool: Whether to use symmetric colourbar limits.
    """
    # A summed group is real but is still a form factor, so it must not fall through to the
    # transition density branch below.
    if already_squared:
        if mode != "modsq":
            raise ValueError("An intensity has no Re or Im amplitude; select modsq.")
        return data, r"Mean intensity", "viridis", False

    if is_transition_density:
        # We just plot the transition density directly.
        plot_data = data
        label = r"$\Phi^{g\to s}(\mathbf{r})$ [Å$^{-3}$]"
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


def apply_plot_limits(coord1, coord2, plane_data, args, plane, momentum_space=True):
    """
    Apply the plot range limits specified in args.
    
    # Arguments:
    - coord1::np.ndarray: First coordinate meshgrid.
    - coord2::np.ndarray: Second coordinate meshgrid.
    - plane_data::np.ndarray: Data to plot.
    - args::argparse.Namespace: Command line arguments.
    - plane::str: Which plane is being plotted.
    - momentum_space::bool: Whether the coordinates are momenta rather than positions.
    
    # Returns:
    - coord1_limited::np.ndarray: Limited first coordinate meshgrid.
    - coord2_limited::np.ndarray: Limited second coordinate meshgrid.
    - plane_data_limited::np.ndarray: Limited data.
    - xlim::tuple: x-axis limits for plotting.
    - ylim::tuple: y-axis limits for plotting.
    """
    # Pick the limits from the physical coordinates, including real-valued intensities.
    if not momentum_space:
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


def get_top_f_sq_lm_modes(f_sq_lm, q_grid, top_n=8):
    """
    Get the dominant f^2_{lm}(q) modes ranked by integrated absolute area.

    # Arguments:
    - f_sq_lm::np.ndarray: The f_sq_lm tensor with shape (n_q, n_keys).
    - q_grid::np.ndarray: The |q| grid in keV.
    - top_n::int: Number of top modes to return (default: 8).

    # Returns:
    - top_modes::list: The dominant (l, m) modes.
    """
    n_keys = f_sq_lm.shape[1]
    l_max = int(np.sqrt(n_keys)) - 1
    areas = {}
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            key = l * l + (l + m) + 1
            areas[(l, m)] = np.trapezoid(np.abs(f_sq_lm[:, key - 1]), q_grid)
    top_modes = sorted(areas, key=areas.get, reverse=True)[:top_n]
    return top_modes

def build_f_sq_lm_style_map(modes):
    """
    Build a consistent line-style map for a collection of f^2_{lm}(q) modes.
    This ensures that modes keep a consistent style between individual conformers and crystal averages. 

    # Arguments:
    - modes::list: The (l, m) modes to assign styles to.

    # Returns:
    - style_map::dict: Mapping from (l, m) to plotting style.
    """
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    style_map = {}

    for idx, mode in enumerate(sorted(set(modes))):
        style_map[mode] = {
            "color": colors[idx % len(colors)],
            "linestyle": linestyles[idx % len(linestyles)],
        }

    return style_map


def plot_f_sq_lm_panel(ax, f_sq_lm, q_grid, title=None, top_n=8, top_modes=None, style_map=None,
                   ylabel=None):
    """
    Plot the top N f^2_{lm}(q) modes on a single set of axes.

    # Arguments:
    - ax::matplotlib.axes.Axes: The axes to plot on.
    - f_sq_lm::np.ndarray: The f_sq_lm tensor with shape (n_q, n_keys).
    - q_grid::np.ndarray: The |q| grid in keV.
    - title::str or None: Optional panel title.
    - top_n::int: Number of top modes to plot, ranked by integrated absolute area (default: 8).
    - top_modes::list or None: Optional list of (l, m) modes to plot.
    - style_map::dict or None: Optional style map keyed by (l, m).
    - ylabel::str or None: Optional y-axis label.
    """
    # Find the top modes if not provided.
    if top_modes is None:
        top_modes = get_top_f_sq_lm_modes(f_sq_lm, q_grid, top_n=top_n)
    if style_map is None:
        style_map = build_f_sq_lm_style_map(top_modes)

    # Plot each of the top modes.
    for l, m in sorted(top_modes):
        key = l * l + (l + m) + 1
        style = style_map[(l, m)]
        label = rf"$({l},\,{m})$"
        ax.plot(q_grid, f_sq_lm[:, key - 1], lw=1.2, ls=style["linestyle"], color=style["color"], label=label)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$q\ [\mathrm{keV}]$")
    ax.set_ylabel(ylabel or r"$f^2_{\ell m}(q)$")
    ax.set_xlim(q_grid[0], q_grid[-1])
    ax.legend(fontsize=8, ncol=2, title=r"Dominant modes: $(\ell,\,m)$", title_fontsize=8)
    if title is not None:
        ax.set_title(title)


def plot_f_sq_lm_modes(f_sq_lm, q_grid, output_dir, top_n=8, ylabel=None):
    """
    Plot the top N f^2_{lm}(q) modes by integrated absolute area on a single set of axes.
    Lines are grouped by l (same colour) with distinct linestyles for different m.

    # Arguments:
    - f_sq_lm::np.ndarray: The f_sq_lm tensor with shape (n_q, n_keys).
    - q_grid::np.ndarray: The |q| grid in keV.
    - output_dir::Path: Directory to save the output plot.
    - top_n::int: Number of top modes to plot, ranked by integrated absolute area (default: 8).
    - ylabel::str or None: Optional y-axis label.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_f_sq_lm_panel(ax, f_sq_lm, q_grid, top_n=top_n, ylabel=ylabel)
    fig.tight_layout()

    fig.savefig(output_dir / "f_sq_lm_modes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_crystal_form_factors_modes(f_sq_lm, conformer_f_sq_lm, conformer_titles, q_grid, output_dir, top_n=8,
                           ylabel=None):
    """
    Plot the aggregate crystal f^2_{lm}(q) modes together with each conformer contribution.

    # Arguments:
    - f_sq_lm::np.ndarray: The aggregate crystal f_sq_lm tensor with shape (n_q, n_keys).
    - conformer_f_sq_lm::np.ndarray: The conformer tensors with shape (n_conformers, n_q, n_keys).
    - conformer_titles::list: The conformer panel titles.
    - q_grid::np.ndarray: The |q| grid in keV.
    - output_dir::Path: Directory to save the output plot.
    - top_n::int: Number of top modes to plot per panel, ranked by integrated absolute area (default: 8).
    - ylabel::str or None: Optional y-axis label for the aggregate crystal panel.
    """
    # Find the top modes for the whole crystal and each conformer.
    crystal_top_modes = get_top_f_sq_lm_modes(f_sq_lm, q_grid, top_n=top_n)
    conformer_top_modes = [get_top_f_sq_lm_modes(conformer_f_sq_lm[i], q_grid, top_n=top_n) for i in range(len(conformer_titles))]
    all_top_modes = list(crystal_top_modes)
    for top_modes in conformer_top_modes:
        for mode in top_modes:
            if mode not in all_top_modes:
                all_top_modes.append(mode)
    # Build the style map based on all modes, too keep it consistent between panels.
    style_map = build_f_sq_lm_style_map(all_top_modes)

    n_panels = 1 + len(conformer_titles)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5), squeeze=False, sharex=True)

    plot_f_sq_lm_panel(axes[0, 0], f_sq_lm, q_grid, title="Crystal aggregate", top_n=top_n,
                   top_modes=crystal_top_modes, style_map=style_map, ylabel=ylabel)
    for panel_idx, conformer_title in enumerate(conformer_titles, start=1):
        plot_f_sq_lm_panel(
            axes[0, panel_idx],
            conformer_f_sq_lm[panel_idx - 1],
            q_grid,
            title=conformer_title,
            top_n=top_n,
            top_modes=conformer_top_modes[panel_idx - 1],
            style_map=style_map,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "f_sq_lm_modes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scattering_rates(rates_dir):
    """
    Plot the DM scattering rates as a function of DM mass.

    # Arguments:
    - rates_dir::Path: Directory containing the scattering_rates HDF5 file.
    """
    input_path = None
    for candidate in [
        rates_dir / "scattering_rates_f64.h5",
        rates_dir / "scattering_rates_f32.h5",
    ]:
        if candidate.exists():
            input_path = candidate
            break

    if input_path is None:
        print("  Warning: scattering rates not found. Was COMPUTE_RATES=true?")
        return False

    with h5py.File(input_path, "r") as rate_file:
        m_chi = rate_file["m_chi_MeV"][()]
        rate_max = rate_file["rate_max"][()]
        rate_min = rate_file["rate_min"][()]
        rate_mean = rate_file["rate_mean"][()]

    fig, ax = plt.subplots(figsize=(7, 4))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.plot(m_chi, rate_max, lw=1.2, ls="-",  color=colors[0], label=r"Maximum over sampled orientations")
    ax.plot(m_chi, rate_min, lw=1.2, ls="--", color=colors[1], label=r"Minimum over sampled orientations")
    ax.plot(m_chi, rate_mean, lw=1.2, ls="-.", color=colors[2], label=r"Mean over sampled orientations")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_\chi\ [\mathrm{MeV}]$")
    ax.set_ylabel(r"Dimensionless rate")
    ax.set_xlim(m_chi[0], m_chi[-1])
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()

    fig.savefig(rates_dir / "scattering_rates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    # Parse the command line arguments.
    args = parse_cli_args()
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    runs_dir = project_root / "runs" / args.run_name / str(args.molecule_number)
    # Non-coherent runs keep the historical default of just the first transition.
    if args.transition_indices is None:
        args.transition_indices = "1"
    transitions_to_plot = [t.strip() for t in args.transition_indices.split(",")]

    # The slices need the (q, θ, ϕ) grid. The f_sq_lm modes and the bands do not.
    slices_available = not args.crystal or getattr(args, "has_grid", True)
    if not slices_available:
        print("No f_uc or I_uc grid in the crystal output, so there are no slices to plot. Re-run with "
              "--compute-mode including form_factor if the group slice plots are wanted.")

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

    crystal_mode = args.crystal_mode
    results_root = args.results_root

    def get_transition_dir(tidx):
        numeric_dir = results_root / str(tidx)
        if numeric_dir.exists():
            return numeric_dir

        transition_dir = results_root / f"transition_{tidx}"
        if transition_dir.exists():
            return transition_dir

        if crystal_mode:
            return numeric_dir
        return transition_dir

    def get_rates_dir():
        return results_root

    # Helper to plot one transition without restarting Python.
    def plot_transition(tidx: str):
        """
        Plot slices for a single transition. Keeps plotting in-process to avoid repeated Python start-up.

        # Arguments:
        - tidx::str: Transition index to plot.
        """
        if args.crystal:
            # One file holds every crystal state, so tidx selects a slice rather than a directory.
            input_path = args.crystal_path
            output_dir = args.crystal_path.parent / (
                f"group_{tidx}" if args.grouped else f"state_{tidx}")
        else:
            tdir = get_transition_dir(tidx)
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
                if (args.crystal and any(key in ff_file for key in ("f_uc", "I_uc"))) or (not args.crystal and "f_s" in ff_file):
                    theta_grid = ff_file["theta_grid"][()]
                    phi_grid = ff_file["phi_grid"][()]

                    if args.crystal:
                        # Julia writes (n_states, n_q, n_theta, n_phi), which HDF5 reverses to
                        # (n_phi, n_theta, n_q, n_states). Take the group, then match the layout below.
                        group = int(tidx)
                        if not 1 <= group <= args.n_groups:
                            print(f"  Group {group} is out of range, the run has {args.n_groups}.")
                            return []
                        first, last = args.group_bounds[group - 1]
                        f_s = crystal_group_grid(ff_file, args.crystal_order, first, last, args.n_molecules)
                    else:
                        f_s = ff_file["f_s"][()]

                    # Julia writes f_s with shape (n_q, n_theta, n_phi).
                    # Due to column-major/row-major differences, HDF5 reverses dimensions.
                    # Python reads as (n_phi, n_theta, n_q), so transpose to (n_theta, n_phi, n_q).
                    f_s = np.transpose(f_s, (1, 0, 2))

                # Load f_sq_lm if requested, and transpose because of the usual Julia vs Python nonsense.
                f_sq_lm = None
                conformer_f_sq_lm = None
                conformer_titles = None
                if args.plot_f_sq_lm_modes and args.crystal:
                    # Average the plotted group, matching the slice above. Incoherent entries just divided by N instad.
                    first, last = args.group_bounds[int(tidx) - 1]
                    f_sq_lm = ff_file["f_sq_lm"][:, :, first:last].mean(axis=2).T
                    if args.crystal_order == "incoherent":
                        f_sq_lm = f_sq_lm / args.n_molecules
                elif args.plot_f_sq_lm_modes and "f_sq_lm" in ff_file:
                    f_sq_lm = ff_file["f_sq_lm"][()].T
                elif args.plot_f_sq_lm_modes:
                    print(f"  Warning: f_sq_lm not found for transition {tidx}. Was f_sq_lm_tensor included in COMPUTE_MODES?")
                
            elif args.method == "fft":
                # Load form factor data.
                form_factor = ff_file["form_factor"][()]
                qx_grid = ff_file["qx_grid"][()]
                qy_grid = ff_file["qy_grid"][()]
                qz_grid = ff_file["qz_grid"][()]

                # Compute q_lim from the grid.
                q_lim = (qx_grid, qy_grid, qz_grid)

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
                q_lim = (qx_grid, qy_grid, qz_grid)

                # Julia writes with shape (nx, ny, nz), but HDF5 reverses to (nz, ny, nx).
                # Transpose to get (nx, ny, nz).
                form_factor = np.transpose(form_factor, (2, 1, 0))

        # Return data needed for plotting without re-reading from disk.
        if args.method == "spherical":
            return [
                f_s,
                theta_grid,
                phi_grid,
                q_grid,
                output_dir,
                f_sq_lm if args.plot_f_sq_lm_modes else None,
                conformer_f_sq_lm if args.plot_f_sq_lm_modes else None,
                conformer_titles if args.plot_f_sq_lm_modes else None,
            ]
        elif args.method == "fft":
            return [None, None, None, None, form_factor, q_lim, output_dir, transition_density if args.plot_transition_density else None, r_lim if args.plot_transition_density else None]
        else:
            return [None, None, None, None, form_factor, q_lim, output_dir]

    # Decide which plots to generate.
    plots_to_generate = []

    if args.method == "spherical":
        if slices_available:
            plots_to_generate.append(("form_factor", False))
        if args.plot_f_sq_lm_modes:
            plots_to_generate.append(("f_sq_lm_modes", False))
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
        if plot_scattering_rates(get_rates_dir()):
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

            # Plot the f_sq_lm^2 coefficients.
            elif plot_type == "f_sq_lm_modes":
                if run_outputs[5] is not None:
                    f_sq_lm_ylabel = None
                    if args.crystal:
                        first, last = args.group_bounds[int(tidx) - 1]
                        f_sq_lm_ylabel = crystal_quantity_label(args.crystal_order, first, last,
                            args.n_molecules, args.original_transitions, coefficients=True)
                    if run_outputs[6] is not None and run_outputs[7] is not None:
                        plot_crystal_form_factors_modes(run_outputs[5], run_outputs[6], run_outputs[7],
                                               run_outputs[3], output_dir, ylabel=f_sq_lm_ylabel)
                    else:
                        plot_f_sq_lm_modes(run_outputs[5], run_outputs[3], output_dir,
                                       ylabel=f_sq_lm_ylabel)
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
                    already_squared = (args.crystal_order is not None
                                       and not np.iscomplexobj(plane_data))
                    plot_data, cbar_label, cmap, symmetric = extract_domain_data(
                        # The incoherent grid holds |f|^2 already.
                        plane_data, mode, already_squared=already_squared, is_transition_density=is_transition_density
                    )

                    if args.crystal:
                        first, last = args.group_bounds[int(tidx) - 1]
                        cbar_label = crystal_quantity_label(args.crystal_order, first, last,
                            args.n_molecules, args.original_transitions, mode=mode)

                    # Apply the plot limits.
                    coord1, coord2, plot_data, xlim, ylim = apply_plot_limits(
                        coord1,
                        coord2,
                        plot_data,
                        args,
                        plane,
                        momentum_space=not is_transition_density,
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
                    fixed_coordinate = {"xy": "z", "xz": "y", "yz": "x"}[plane]
                    coordinate = "" if is_transition_density else "q_"
                    fixed_value = 0.0
                    if args.method != "spherical":
                        fixed_axis = {"xy": 2, "xz": 1, "yz": 0}[plane]
                        if is_transition_density:
                            half_width = run_outputs[8][fixed_axis]
                            points = run_outputs[7].shape[fixed_axis]
                            fixed_grid = np.linspace(-half_width, half_width, points, endpoint=False)
                        else:
                            fixed_grid = run_outputs[5][fixed_axis]
                        fixed_value = fixed_grid[np.argmin(np.abs(fixed_grid))]
                    units = "Å" if is_transition_density else "keV"
                    ax.set_title(rf"${coordinate}{fixed_coordinate}={fixed_value:g}$ ({units})")

                    # Add the colourbar for each subplot.
                    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
                    cbar.set_label(cbar_label, rotation=270, labelpad=32)

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

    # The bands live beside the coherent output, written by the form factor run when it was asked
    # for a band map. They are not slices of q, so they get their own figure rather than a panel.
    if args.crystal and args.crystal_path is not None:
        with h5py.File(args.crystal_path, "r") as crystal_file:
            has_bands = "band_map" in crystal_file
            has_structure_factor = any(key in crystal_file for key in ("S_lm",))
        if has_bands:
            plot_band_energies(args.crystal_path, args.crystal_path.parent / "bands")
        elif args.crystal_order == "coherent":
            print("No band_map group in the coherent output, so no band energy figures. Re-run the "
                  "form factor with --band-map-plane to get them.")

        # Automatically plot the structure factor if it exists.
        if has_structure_factor:
            plot_structure_factor(args.crystal_path,
                                  args.crystal_path.parent / "structure_factor")
        else:
            print("No S_lm in the crystal output, so no (q, E) figures. Re-run "
                  "the form factor without --no-structure-factor to get them.")


if __name__ == "__main__":
    main()
