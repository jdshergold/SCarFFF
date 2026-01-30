import argparse
import csv
import time

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
from pyscf import gto, dft, tddft
from pyscf.tools import cubegen
from pyscf.geomopt.geometric_solver import optimize
import plotly.graph_objects as go
import h5py
import numpy as np
import os
from sklearn.cluster import MeanShift
import warnings
from contextlib import redirect_stdout, redirect_stderr

# Atomic numbers for elements up to Mo, used for weighting when flattening molecule.
ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
}

BOHR_TO_ANGSTROM = 0.52917721092

def get_rdkit_optimised_geometry(smiles="C1=CC=CC=C1"):
    """Use RDKit to generate quasi-optimal molecular geometry for a given SMILES string.

    # Arguments:
    - smiles::str: SMILES string representing the molecule (default: benzene).

    # Returns:
    - coordinates::list: List of atom symbols and their coordinates.
    - ring_indices::tuple: Tuple of ring atom indices (if any).
    """

    # Create molecule from SMILES string.
    molecule = Chem.MolFromSmiles(smiles)

    # Add hydrogen atoms.
    molecule = Chem.AddHs(molecule)

    # Set up embedding parameters with pruning to avoid conformers that are too similar.
    params = AllChem.ETKDGv3()
    params.randomSeed = 79911030
    params.numThreads = 0  # Use all threads.
    params.pruneRmsThresh = (
        0.3  # Drop any conformers that are the same to within 0.3 Angstrom.
    )

    # Generate the conformers.
    cids = AllChem.EmbedMultipleConfs(molecule, numConfs=30, params=params)

    # Check that we have at least one valid conformer, otherwise fall back to random coords and no basic knowledge.
    # This should hopefully only come up for very large or unusual molecules, such as fullerenes.
    if list(cids) == []:
        print("No conformers generated, retrying with relaxed parameters.")
        params.useRandomCoords = True
        params.useBasicKnowledge = False
        params.useExpTorsionAnglePrefs = True
        params.useSmallRingTorsions = True
        cids = AllChem.EmbedMultipleConfs(molecule, numConfs=30, params=params)

    # Optimise all of the conformers and find the one with lowest energy.
    results = AllChem.UFFOptimizeMoleculeConfs(molecule, numThreads=0)

    energies = [res[1] for res in results]
    best_idx = int(np.argmin(energies))
    best_cid = int(cids[best_idx])

    # Extract the coordinates from the best conformer.
    best_conf = molecule.GetConformer(best_cid)

    coordinates = []
    for i in range(molecule.GetNumAtoms()):
        atom = molecule.GetAtomWithIdx(i)
        pos = best_conf.GetAtomPosition(i)
        coordinates.append([atom.GetSymbol(), pos.x, pos.y, pos.z])

    # Get the ring information.
    ring_indices = molecule.GetRingInfo().AtomRings()

    return coordinates, ring_indices

def get_dft_optimised_geometry(coords, basis="6-31g*", use_gpu=False):
    """
    Use PySCF to generate optimised molecular geometry for a given SMILES string.

    # Arguments:
    - coords::list: List of atom symbols and their coordinates, which should already be optimised by RDKit.
    - basis::str: Basis set for the calculation (default: '6-31g*').
    - use_gpu::bool: Whether to run the DFT calculation on a GPU (default: False).

    # Returns:
    - coordinates::list: List of atom symbols and their coordinates.
    """

    # Create the PySCF molecule.
    mol = create_pyscf_mol(coords, basis=basis)
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf = mf.density_fit()

    # Silence this part. It's very noisy. This includes error messages, which aren't actually errors from geometric.
    mf.verbose = 0
    mf.mol.verbose = 0

    print("Running geometric optimisation with PySCF.")
    if use_gpu:
        print("Moving mean-field object to GPU for geometry optimisation...")
        mf = mf.to_gpu()

    # Run the geometry optimisation. Be quiet.
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        mol_eq = optimize(mf)

    if use_gpu:
        print("Moving mean-field object back to CPU after geometry optimisation...")
        mf = mf.to_cpu()
    
    # Extract the optimised coordinates and convert back to Angstrom from Bohr.
    # Note that PySCF returns coordinates in Bohr units, whereas RDKit uses Angstrom. This is why we don't convert the input coords/ones that do not use the DFT optimisation.
    coordinates = [[atom[0], atom[1][0] * BOHR_TO_ANGSTROM, atom[1][1] * BOHR_TO_ANGSTROM, atom[1][2] * BOHR_TO_ANGSTROM] for atom in mol_eq._atom]

    return coordinates


def flatten_molecule_ring(coordinates, ring_indices):
    """
    Flatten the molecule based on any rings. First find the vectors perpendicular to each ring plane,
    then rotate the average vector onto the z-axis. Then rotate the molecule such that the first principal
    direction in the xy-plane points along the x-axis.

    # Arguments:
    - coordinates::list: List of atom coordinates and their symbols.
    - ring_indices::tuple: Tuple of ring atom indices (if any).

    # Returns:
    - flattened_coordinates::list: List of flattened atom coordinates and their symbols.
    """

    # If no rings, just flatten based on whole molecule.
    if len(ring_indices) == 0:
        print("No rings found, flattening based on whole molecule.")
        return flatten_molecule(coordinates)

    # Separate the symbols and coordinates.
    symbols = [atom[0] for atom in coordinates]
    coordinates = np.array([[atom[1], atom[2], atom[3]] for atom in coordinates])
    weights = np.array([ATOMIC_NUMBERS[symbol] for symbol in symbols])

    centred_coords = coordinates - np.average(coordinates, axis=0, weights=weights)

    normals = []
    ring_weights = [
        sum(ATOMIC_NUMBERS[symbols[i]] for i in ring) for ring in ring_indices
    ]
    # Compute the normal vectors for each ring using PCA.
    for ring in ring_indices:
        ring_coords = centred_coords[np.array(ring)]
        cov_matrix = np.cov(ring_coords, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors. The [::-1] ensures that they will be sorted in descending order.
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # The normal vector is the eigenvector corresponding to the smallest eigenvalue.
        normal = eigenvectors[:, -1]
        normals.append(normal)

    # Find the weighted average normal vector.
    avg_normal = np.average(normals, axis=0, weights=ring_weights)

    # Rotate the average normal vector onto the z-axis.
    rperp = np.sqrt(avg_normal[0] ** 2 + avg_normal[1] ** 2)  # rperp = sqrt(x^2 + y^2).
    r = np.sqrt(avg_normal[2] ** 2 + rperp**2)  # r = sqrt(z^2 + rperp^2).

    if rperp == 0:
        # The average normal vector is already parallel to the z-axis.
        R = np.eye(3)
    else:
        R = np.array(
            [
                [
                    avg_normal[0] * avg_normal[2] / (r * rperp),
                    avg_normal[1] * avg_normal[2] / (r * rperp),
                    -rperp / r,
                ],
                [-avg_normal[1] / rperp, avg_normal[0] / rperp, 0],
                [avg_normal[0] / r, avg_normal[1] / r, avg_normal[2] / r],
            ]
        )

    # Apply the rotation to the centred coordinates.
    rotated_coords = [R @ coord for coord in centred_coords]

    # Now find the principal directions in the xy-plane.
    rotated_coords = np.array(rotated_coords)
    cov_matrix = np.cov(rotated_coords[:, :2], rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors. The [::-1] ensures that they will be sorted in descending order.
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    v1 = eigenvectors[:, 0]

    # Now find the matrix rotates the first principal direction onto the x-axis (but in 3D).
    r = np.sqrt(v1[0] ** 2 + v1[1] ** 2)  # r = sqrt(x1^2 + y1^2).

    if r == 0:
        # The first principal direction is already parallel to the x-axis.
        R2 = np.eye(3)
    else:
        R2 = np.array(
            [[v1[0] / r, v1[1] / r, 0], [-v1[1] / r, v1[0] / r, 0], [0, 0, 1]]
        )

    # Apply the second rotation to the rotated coordinates.
    flattened_coordinates = [R2 @ coord for coord in rotated_coords]
    flattened_coordinates = [
        [
            symbols[i],
            flattened_coordinates[i][0],
            flattened_coordinates[i][1],
            flattened_coordinates[i][2],
        ]
        for i in range(len(symbols))
    ]

    return flattened_coordinates


def flatten_molecule(coordinates):
    """
    Shift the mean of the molecule coordinates to the origin, and then rotate such that
    the first principal direction points along the x-axis, and the second principal direction
    points along the y-axis.

    # Arguments:
    - coordinates::list: List of atom coordinates and their symbols.

    # Returns:
    - flattened_coordinates::list: List of flattened atom coordinates and their symbols.
    """

    # Separate the symbols and coordinates.
    symbols = [atom[0] for atom in coordinates]
    coordinates = np.array([[atom[1], atom[2], atom[3]] for atom in coordinates])
    weights = np.array([ATOMIC_NUMBERS[symbol] for symbol in symbols])

    # Compute the weighted mean of the coordinates.
    mean = np.average(coordinates, axis=0, weights=weights)

    # Centre both the active coordinates and full set of coordinates.
    centred_coords = coordinates - mean

    # Compute the covariance matrix. Rowvar saves us transposing the input.
    cov_matrix = np.cov(centred_coords, rowvar=False)

    # Compute the eigenvalues and eigenvectors.
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors. The [::-1] ensures that they will be sorted in descending order.
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Construct the matrix that rotates the first principal direction onto the x-axis.
    v1, v2 = eigenvectors[:, 0], eigenvectors[:, 1]

    rperp1 = np.sqrt(v1[1] ** 2 + v1[2] ** 2)  # rperp1 = sqrt(y1^2 + z1^2).
    r1 = np.sqrt(v1[0] ** 2 + rperp1**2)  # r1 = sqrt(x1^2 + rperp1^2).

    if rperp1 == 0:
        # The first principal direction is already parallel to the x-axis.
        R1 = np.eye(3)
    else:
        R1 = np.array(
            [
                [v1[0] / r1, v1[1] / r1, v1[2] / r1],
                [0, -v1[2] / rperp1, v1[1] / rperp1],
                [
                    rperp1 / r1,
                    -v1[0] * v1[1] / (r1 * rperp1),
                    -v1[0] * v1[2] / (r1 * rperp1),
                ],
            ]
        )

    # Apply the rotation R1 to the second principal direction.
    v2_rot = R1 @ v2
    rperp2 = np.sqrt(v2_rot[1] ** 2 + v2_rot[2] ** 2)  # rperp2 = sqrt(y2'^2 + z2'^2).

    # This now lives in the yz-plane. We just need to rotate about the x-axis to align it with y.
    R2 = np.array(
        [
            [1, 0, 0],
            [0, v2_rot[1] / rperp2, v2_rot[2] / rperp2],
            [0, -v2_rot[2] / rperp2, v2_rot[1] / rperp2],
        ]
    )

    # Find the combined rotation matrix.
    Rtot = R2 @ R1

    # Apply the rotation to the centred coordinates, and add back the symbols.
    flattened_coordinates = [Rtot @ coord for coord in centred_coords]
    flattened_coordinates = [
        [
            symbols[i],
            flattened_coordinates[i][0],
            flattened_coordinates[i][1],
            flattened_coordinates[i][2],
        ]
        for i in range(len(symbols))
    ]

    return flattened_coordinates

def plot_molecule_3d_interactive(coordinates, output_directory="."):
    """Save the atomic positions plot to an HTML file using Plotly.

    # Arguments:
    - coordinates::list: List of atom symbols and their coordinates.
    - output_directory::str: Directory path to save the plot (default: ".").
    """

    # Extract data.
    symbols = [atom[0] for atom in coordinates]
    x = [atom[1] for atom in coordinates]
    y = [atom[2] for atom in coordinates]
    z = [atom[3] for atom in coordinates]

    # Colour and size mapping.
    colours = ["black" if s == "C" else "lightgray" for s in symbols]
    sizes = [15 if s == "C" else 8 for s in symbols]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+text",
                marker=dict(size=sizes, color=colours),
                text=symbols,
                textposition="middle right",
            )
        ]
    )

    fig.update_layout(
        title="Molecule",
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
    )

    fig.write_html(f"{output_directory}/molecule_3d.html")
    print(f"3D molecule plot saved to {output_directory}/molecule_3d.html.")


def create_pyscf_mol(coordinates, basis="ccpvdz"):
    """Create a PySCF molecule object with the optimised geometry.

    # Arguments:
    - coordinates::list: List of atom symbols and their coordinates.
    - basis::str: Basis set for the calculation (default: 'ccpvdz').

    # Returns:
    - mol::Mole: The PySCF molecule object.
    """

    mol = gto.Mole()
    mol.atom = coordinates
    mol.basis = basis
    mol.symmetry = False  # True
    mol.build()
    return mol


def print_excited_states(td):
    """Print information about excited states from TD-DFT calculation.

    # Arguments:
    - td::TDDFT: The TD-DFT object containing excited state information.
    """
    print("\nExcited State Energies:")
    print("-" * 50)
    print("State   Energy (eV)   Oscillator Strength   Major Transitions")
    print("-" * 50)

    for i, (energy, oscillator_strength) in enumerate(
        zip(td.e, td.oscillator_strength())
    ):
        energy_ev = energy * 27.211396  # Convert from Hartree to eV.
        print(f"{i+1:3d}     {energy_ev:8.4f}      {oscillator_strength:8.4f}")


def calculate_excited_states(mol, nstates=10, use_gpu=False):
    """Run TD-DFT calculation to find excited states.

    # Arguments:
    - mol::Mole: The PySCF molecule object.
    - nstates::Int: The number of excited states to calculate (default: 10).
    - use_gpu::bool: Whether to run the DFT and TD-DFT calculations on a GPU (default: False).
    """

    # First run DFT calculation with B3LYP functional.
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"

    # Move to GPU if requested.
    if use_gpu:
        print("Moving mean-field object to GPU...")
        mf = mf.to_gpu()

    mf.kernel()

    # Move back to CPU if GPU was used.
    if use_gpu:
        print("Moving mean-field object back to CPU...")
        mf = mf.to_cpu()

    # Check if SCF converged.
    if not mf.converged:
        print("DFT calculation did not converge!")
        return None, None

    # Run TD-DFT calculation for excited states.
    td = tddft.TDDFT(mf)
    td.nstates = nstates

    # Move to GPU if requested.
    if use_gpu:
        print("Moving TD-DFT object to GPU...")
        td = td.to_gpu()

    td.kernel()

    # Move back to CPU if GPU was used.
    if use_gpu:
        print("Moving TD-DFT object back to CPU...")
        td = td.to_cpu()

    # Print results.
    print_excited_states(td)

    return mf, td


def write_nto(mol, ID, wghts, nto):
    """Write out the dominant occupied/virtual NTO pair in cube format for visualization/portability.

    # Arguments:
    - mol::Mole: The PySCF molecule object.
    - ID::str: Identifier for the NTO pair.
    - wghts::np.ndarray: Weights of the NTOs.
    - nto::np.ndarray: NTO coefficients.
    """
    nocc = len(wghts)
    cubegen.orbital(mol, ID + "occ.cube", nto[:, 0])
    cubegen.orbital(mol, ID + "vir.cube", nto[:, nocc])


def get_tdm(tdobj, state=1):
    """
    Returns the transition density matrix in the MO basis. This is constructed from
    the TDDFT X and Y matrices as T = X + Y, and should already be normalised.

    # Arguments:
    - tdobj::TDDFT: The TD-DFT object containing excited state information.
    - state::int: The excited state to analyze (default: 1).

    # Returns:
    - cis_t1::np.ndarray: The transition density matrix in the MO basis.
    """
    state_id = state - 1
    X = tdobj.xy[state_id][0]
    Y = tdobj.xy[state_id][1]

    # PySCF returns eigenvectors that are already normalised, so keep both
    # excitation (X) and de-excitation (Y) components to preserve magnitude.
    TDM = X + Y

    # Factor of sqrt(2) for spin degeneracy.
    return np.sqrt(2) * TDM


def run_td_dft_analysis(
    smiles="C1=CC=CC=C1",
    basis="6-31g*",
    nstates=5,
    ntrans=3,
    output_directory=".",
    ring_flatten=True,
    use_gpu=False,
    dft_optimisation=False,
    precision="float64",
    plot_molecule_3d=False,
):
    """Perform the TD-DFT analysis for a given molecule and save the results to an HDF5 file.

    Arguments:
        smiles::str: SMILES string representing the molecule (default: benzene).
        basis::str: Basis set for the calculation (default: '6-31g*').
        nstates::int: Number of excited states to calculate (default: 5).
        ntrans::int: Number of transitions to compute transitiom matrices for (default: 3).
        output_directory::str: Directory path to save the results (default: '.').
        ring_flatten::bool: Whether to flatten based on just ring coordinates, or the whole molecule.
        dft_optimisation::bool: Whether to use DFT for geometry optimisation after RDKit (default: False).
        use_gpu::bool: Whether to run the DFT and TD-DFT calculations on a GPU (default: False).
        precision::str: Floating point precision to use when saving results ('float32' or 'float64').
        plot_molecule_3d::bool: Whether to generate the interactive 3D molecule plot (default: False).

    Returns:
        tuple: Contains coordinates, molecule object, mean-field object, TD-DFT object, and d_ij matrices.
    """
    print("Optimising geometry with RDKit.")
    coords0, ring_indices = get_rdkit_optimised_geometry(smiles=smiles)

    if dft_optimisation:
        print("Further optimising geometry with PySCF DFT.")
        coords = get_dft_optimised_geometry(coords0, basis=basis, use_gpu=use_gpu)
    else:
        coords = coords0

    print("Flattening molecule.")
    if ring_flatten:
        coords = flatten_molecule_ring(coords, ring_indices=ring_indices)
    else:
        coords = flatten_molecule(coords)

    if plot_molecule_3d:
        plot_molecule_3d_interactive(coords, output_directory)
    print("RDKit optimisation complete.")

    print("Creating pySCF molecule with generated geometry.")
    mol = create_pyscf_mol(coords, basis=basis)
    print("Molecule created.")

    print("Running TD-DFT calculation for excited states.")
    mf, td = calculate_excited_states(mol, nstates=nstates, use_gpu=use_gpu)
    if mf is None:
        print("Calculation failed. Exiting.")
        return coords, mol, None, None, None
    else:
        td.analyze()

    print(f"Computing d_ij for the first N={ntrans} transitions.")

    # Extract the TDM for the first ntrans transitions.
    mxTDM = [get_tdm(td, state=i + 1) for i in range(ntrans)]

    # Find which MOs are occupied and which are virtual.
    occupations = mf.mo_occ
    occ_mask = occupations > 0

    # Extract the molecular orbital coefficients.
    orbo = mf.mo_coeff[:, occ_mask]
    orbv = mf.mo_coeff[:, ~occ_mask]

    # Use these alongside the TDM to construct d_{ij}.
    mxDij = [orbo @ tdm @ (orbv.conj().T) for tdm in mxTDM]

    # Prepare data for saving to file.
    geometry = [[atom[0], atom[1][0], atom[1][1], atom[1][2]] for atom in mol._atom]
    ao_labels = [str(label) for label in mol.ao_labels()]
    energies = [float(energy * 27.211396) for energy in td.e]  # Convert Hartree to eV.

    # Extract oscillator strengths.
    oscillator_strengths = np.array(
        [float(td.oscillator_strength()[i]) for i in range(ntrans)]
    )

    # Set the output precision.
    precision = precision.lower()
    if precision not in ("float32", "float64"):
        raise ValueError(f"Unsupported precision {precision} specified. Suppported types are float32 and float64.")
    float_dtype = np.float32 if precision == "float32" else np.float64
    complex_dtype = np.complex64 if precision == "float32" else np.complex128
    precision_suffix = "_f32" if precision == "float32" else "_f64"

    # Save to HDF5 file with the correct precision suffix.
    output_file = f"{output_directory}/td_dft_results{precision_suffix}.h5"
    with h5py.File(output_file, "w") as f:
        # Store the molecule metadata as attributes.
        f.attrs["species"] = smiles
        f.attrs["basis"] = basis

        # Store the geometry.
        atom_symbols = [atom[0] for atom in geometry]
        atom_coords = np.array([[atom[1], atom[2], atom[3]] for atom in geometry], dtype=float_dtype)

        # Save the atom symbols and coordinates.
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("atom_symbols", data=atom_symbols, dtype=dt)
        f.create_dataset("atom_coordinates", data=atom_coords)

        # Store the AO labels.
        f.create_dataset("ao_labels", data=ao_labels, dtype=dt)

        # Store the energies and oscillator strengths.
        f.create_dataset("energies_ev", data=np.array(energies, dtype=float_dtype))
        f.create_dataset("f_osc", data=np.asarray(oscillator_strengths, dtype=float_dtype))

        # Store the transition matrices.
        for i, dij_matrix in enumerate(mxDij):
            f.create_dataset(f"d_ij_state_{i+1}", data=np.asarray(dij_matrix, dtype=complex_dtype))

    print(f"\nResults saved to {output_file}")

    return coords, mol, mf, td, mxDij


def parse_cli_arguments():
    parser = argparse.ArgumentParser(
        description="Run TD-DFT analysis and write results to HDF5."
    )
    parser.add_argument(
        "--smiles",
        default=None,
        help="SMILES string for a single molecule (ignored if --csv-file is provided).",
    )
    parser.add_argument(
        "--csv-file",
        dest="csv_file",
        default=None,
        help="Path to CSV file containing SMILES strings (one per line, with 'smiles' header).",
    )
    parser.add_argument(
        "--basis",
        default="6-31g*",
        help="Basis set label to use for PySCF.",
    )
    parser.add_argument(
        "--nstates",
        type=int,
        default=5,
        help="Number of excited states to compute.",
    )
    parser.add_argument(
        "--ntrans",
        type=int,
        default=3,
        help="Number of transitions for which to compute NTOs.",
    )
    parser.add_argument(
        "--no-ring-flatten",
        dest="ring_flatten",
        action="store_false",
        help="Disable ring-only flattening (use entire molecule instead).",
    )
    parser.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        help="Enable GPU acceleration for DFT and TD-DFT calculations.",
    )
    parser.add_argument(
        "--precision",
        choices=["float64", "float32"],
        default="float64",
        help="Floating point precision to use when saving TD-DFT results.",
    )
    parser.add_argument(
        "--dft-optimisation",
        dest="dft_optimisation",
        action="store_true",
        help="Perform a DFT geometry optimisation after RDKit.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Output directory for this run (e.g., ../runs/run_name). Molecule subdirectories (1, 2, 3...) will be created inside this. If not specified, defaults to ../runs/{smiles} for single molecule or ../runs/batch_run for CSV.",
    )
    parser.add_argument(
        "--plot-molecule-3d",
        dest="plot_molecule_3d",
        action="store_true",
        help="Generate interactive 3D molecule plot.",
    )
    parser.set_defaults(ring_flatten=True, use_gpu=False, dft_optimisation=False, plot_molecule_3d=False)
    return parser.parse_args()


def main():
    args = parse_cli_arguments()

    # Read the list of molecules from the CSV, or use a single SMILES.
    molecules = []
    if args.csv_file is not None:
        # Construct the path to the CSV file.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        csv_path = os.path.join(project_root, args.csv_file)

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                molecules.append(row['smiles'])
        print(f"Found {len(molecules)} molecules to process.\n")
    elif args.smiles is not None:
        # Single molecule mode.
        molecules = [args.smiles]
        print(f"Processing single molecule with SMILES {args.smiles}\n")
    else:
        raise ValueError("Either --smiles or --csv-file must be provided.")

    # Determine the output directory.
    if args.output_dir is not None:
        run_directory = args.output_dir
    elif args.csv_file is not None:
        run_directory = os.path.join("../runs", "batch_run")
    else:
        run_directory = os.path.join("../runs", args.smiles)

    os.makedirs(run_directory, exist_ok=True)

    # Create the metadata file.
    metadata_file = os.path.join(run_directory, "metadata.csv")
    with open(metadata_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['folder', 'smiles', 'basis'])

    # Start timing the computation.
    computation_start = time.perf_counter()

    # Process each molecule.
    for mol_idx, smiles in enumerate(molecules):
        mol_num = mol_idx + 1  # 1-based numbering for the output directories.
        mol_output_dir = os.path.join(run_directory, str(mol_num))
        os.makedirs(mol_output_dir, exist_ok=True)

        print(f"{'='*50}")
        print(f"Processing molecule {mol_num} of {len(molecules)}: {smiles}")
        print(f"{'='*50}\n")

        # Append to metadata.
        with open(metadata_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([mol_num, smiles, args.basis])

        # Run the TD-DFT analysis for this molecule.
        try:
            run_td_dft_analysis(
                smiles=smiles,
                basis=args.basis,
                nstates=args.nstates,
                ntrans=args.ntrans,
                output_directory=mol_output_dir,
                ring_flatten=args.ring_flatten,
                use_gpu=args.use_gpu,
                dft_optimisation=args.dft_optimisation,
                precision=args.precision,
                plot_molecule_3d=args.plot_molecule_3d,
            )
        except Exception as e:
            # Write a failure marker with the error message for downstream scripts and user diagnostics.
            failed_file = os.path.join(mol_output_dir, ".tddft_failed")
            with open(failed_file, 'w') as f:
                f.write(f"molecule: {smiles}\n")
                f.write(f"error: {e}\n")

            print(f"\nTD-DFT calculation failed for molecule {mol_num} ({smiles}): {e}")
            print(f"Failure recorded in {failed_file}. Skipping molecule {mol_num} and continuing.\n")
            continue

        print(f"\nMolecule {mol_num} complete!\n")

    # End timing and save to file for bash to read.
    computation_time = time.perf_counter() - computation_start
    timing_file = os.path.join(run_directory, ".tddft_time")
    with open(timing_file, 'w') as f:
        f.write(f"{computation_time:.3f}\n")

    print(f"{'='*50}")
    print(f"All TD-DFT calculations complete!")
    print(f"Results saved to: {run_directory}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
