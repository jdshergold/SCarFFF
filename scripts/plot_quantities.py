# Names and normalisations shared by the crystal plots.

import numpy as np


def crystal_grid_key(order):
    """
    Find the coherent unit-cell amplitude or incoherent unit-cell intensity.

    # Arguments:
    - order::str: The crystal order, coherent or incoherent.

    # Returns:
    - str: The grid dataset name.
    """
    return "I_uc" if order == "incoherent" else "f_uc"


def crystal_group_grid(h5_file, order, first, last, n_images):
    """
    Read one band amplitude or a mean intensity over the selected states.

    # Arguments:
    - h5_file::h5py.File: The crystal output file, with the state axis last in Python.
    - order::str: The crystal order, coherent or incoherent.
    - first::int: The first selected state, counted from zero.
    - last::int: One past the last selected state.
    - n_images::int: The number of molecules in the unit cell.

    # Returns:
    - np.ndarray: A single coherent amplitude, or the mean intensity. Incoherent means are per molecule.
    """
    grid = h5_file[crystal_grid_key(order)]
    if order == "coherent" and last - first == 1:
        return grid[..., first]
    intensity = np.zeros(grid.shape[:-1], dtype=float)
    for state in range(first, last):
        intensity += grid[..., state] if order == "incoherent" else np.abs(grid[..., state]) ** 2
    divisor = last - first
    if order == "incoherent":
        divisor *= n_images
    return intensity / divisor


def crystal_quantity_label(order, first, last, n_images, transition_indices, mode="modsq", coefficients=False, html=False):
    """
    Label the selected crystal quantity, including the mean used in the plot.

    # Arguments:
    - order::str: The crystal order, coherent or incoherent.
    - first::int: The first selected state, counted from zero.
    - last::int: One past the last selected state.
    - n_images::int: The number of molecules in the unit cell.
    - transition_indices::array: Original TD-DFT state numbers.
    - mode::str: modsq, Re or Im.
    - coefficients::bool: Whether the plot shows real harmonic intensity coefficients.
    - html::bool: Whether the label is for the interactive plot.

    # Returns:
    - str: The quantity label.
    """
    n_members = last - first
    if order == "incoherent":
        states = ",".join(str(int(s)) for s in transition_indices[first:last])
        divisor = n_images * n_members
        if html:
            return f"(1/{divisor}) Σ<sub>s∈{{{states}}}</sub> Σ<sub>A_i</sub> |f<sub>A_i,s</sub>(q)|²"
        quantity = r"f^2_{A_i,s;\ell m}(q)" if coefficients else r"|f_{A_i,s}(\mathbf{q})|^2"
        return rf"$\frac{{1}}{{{divisor}}}\sum_{{s\in\{{{states}\}}}}\sum_{{A_i}} {quantity}$"
    if html:
        if n_members > 1:
            return f"(1/{n_members}) Σ<sub>n={first+1}…{last}</sub> |f<sub>n,uc</sub>(q)|²"
        quantity = f"f<sub>{first+1},uc</sub>(q)"
        return f"|{quantity}|²" if mode == "modsq" else f"{mode}[{quantity}]"
    quantity = rf"f^2_{{n,\mathrm{{uc}};\ell m}}(q)" if coefficients else r"|f_{n,\mathrm{uc}}(\mathbf{q})|^2"
    if n_members > 1:
        return rf"$\frac{{1}}{{{n_members}}}\sum_{{n={first+1}}}^{{{last}}} {quantity}$"
    if coefficients:
        return rf"$f^2_{{{first+1},\mathrm{{uc}};\ell m}}(q)$"
    amplitude = rf"f_{{{first+1},\mathrm{{uc}}}}(\mathbf{{q}})"
    return rf"$|{amplitude}|^2$" if mode == "modsq" else rf"$\mathrm{{{mode}}}[{amplitude}]$"


def read_structure_function(h5_file):
    """
    Read S_{ell m}(q,E), including 2π/V_cell.

    # Arguments:
    - h5_file::h5py.File: The crystal output file.

    # Returns:
    - tuple: S_lm in Å^-3 eV^-1 with axes (E, q, key), E in eV and q in keV.
    """
    return np.asarray(h5_file["S_lm"]).T, h5_file["E_grid_eV"][()], h5_file["q_grid"][()]
