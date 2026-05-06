import warnings

import numpy as np
from networkx.algorithms import isomorphism
from scipy.spatial.transform import Rotation
from pymatgen.analysis.dimensionality import get_structure_components
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

# Two molecules are the same conformer if their Kabsch RMSD is below this threshold.
CONFORMER_RMSD_THRESHOLD = 0.01  # Å


def kabsch_rotation(x_ref, x_prime):
    """
    Compute the rotation matrix R that best maps x_ref onto x_prime (Kabsch algorithm).

    There's an exact version, but the SVD version is more numerically stable.
    This is just the Kabsch algorithm. It follows from:
    R x ~= x' (here it is exactly equal).
    The goal is to minimise ||R x - x'|| = Tr(x^T x) + Tr(x'^T x') - 2 Tr(x'^T R x),
    or equivalently maximise the last term Tr(x'^T R x) = Tr(R x x'^T).
    Define H = x x'^T, and perform an SVD, H = U S V^T, then also define M = V^T R U, which is orthogonal.
    Then we have to maximise Tr(M S) = sum_k S_{kk} M_{kk}, since S is diagonal.
    Since M is orthogonal, M_{kk} <= 1, and S_{kk} >= 0, the maximum is for the identity M = I.
    This tells us to solve V^T R U = I, which is equivalent to R = V U^T.

    Args:
        x_ref:   (3, N) array — reference coordinates (column vectors).
        x_prime: (3, N) array — coordinates to rotate onto the reference.

    Returns:
        R: (3, 3) rotation matrix such that R @ x_ref ~= x_prime.
    """
    H = x_ref @ x_prime.T
    U, _, Vt = np.linalg.svd(H)
    return Vt.T @ U.T


def parse_cif_structure(path):
    """
    Parse a CIF file into a pymatgen Structure object.

    # Arguments:
    - path::str: Path to the CIF file.

    # Returns:
    - structure::Structure: The parsed structure.
    """
    # Some older CIFs list symmetry-expanded atoms explicitly, such that applying symmetry ops
    # produces doubly-occupied sites (occupancy 2). Retrying with a higher
    # occupancy_tolerance lets pymatgen rescale those back to 1.0.
    for tol in (1.0, 2.1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parser = CifParser(path, occupancy_tolerance=tol)
                structures = parser.parse_structures(primitive=False)
            if structures:
                return structures[0]
        except ValueError:
            continue
    raise ValueError(f"Could not parse CIF: {path}")


def build_coordinate_list(atom_symbols, atom_coordinates):
    """
    Build the coordinate list format expected by the TD-DFT code.

    # Arguments:
    - atom_symbols::list: The atomic symbols.
    - atom_coordinates::list: The Cartesian coordinates in Angstroms.

    # Returns:
    - coordinates::list: List of atom symbols and their coordinates.
    """

    coordinates = []
    for symbol, coord in zip(atom_symbols, atom_coordinates):
        coordinates.append([symbol, coord[0], coord[1], coord[2]])

    return coordinates


def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Convert a proper rotation matrix into a unit quaternion in scalar-first ordering.

    # Arguments:
    - rotation_matrix::np.ndarray: A proper 3x3 rotation matrix.

    # Returns:
    - q::list: The quaternion [w, x, y, z].
    """
    q = Rotation.from_matrix(rotation_matrix).as_quat()
    
    # Scipy returns [x, y, z, w], but we want it in the form [w, x, y, z].
    return [q[3], q[0], q[1], q[2]]             


def get_crystal_data(cif_path):
    """
    Parse a CIF and identify the distinct conformers and their images in the unit cell.

    # Arguments:
    - cif_path::str: Path to the CIF file.

    # Returns:
    - crystal_data::dict: Crystal data for the pipeline.
    """

    structure = parse_cif_structure(cif_path)

    # Read disorder groups from the raw CIF block and build a label -> group mapping.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_block = next(iter(CifParser(cif_path).as_dict().values()))

    site_labels = raw_block["_atom_site_label"]
    site_disorder_groups = raw_block.get("_atom_site_disorder_group", ["."] * len(site_labels)) # Get the disorder group, or use "." if not present.
    site_occupancies = raw_block.get("_atom_site_occupancy", ["1"] * len(site_labels)) # Get the occupancy, or use 1 if not present.
    label_to_group = dict(zip(site_labels, site_disorder_groups))
    group_to_occ = {g: float(o) for l, g, o in zip(site_labels, site_disorder_groups, site_occupancies) if g not in (".", "0")}

    # Tag each site in the fully-expanded structure with its disorder group via its label.
    site_groups = [label_to_group.get(site.label, ".") for site in structure]

    # Find the distinct non-trivial disorder groups.
    unique_disorder_groups = sorted(set(g for g in site_groups if g not in (".", "0"))) or [None]

    # Define the node matching function, which checks that the species are the same.
    node_match = lambda d1, d2: str(d1.get("specie", "")) == str(d2.get("specie", ""))

    conformer_coordinates = {}  # This has structure label:reference_coords,  with reference coords and array of shape (N,3), centred at the origin.
    conformer_species = {}  # This has structure label:element_strings.
    conformer_graphs = {}  # This has structure label:molecule Graph, and is used for isomorphism matching.
    next_label_ord = [ord("A")]  # Ord maps unicode to a number, e.g. "A -> 1, B -> 2". Not actually 1 or 2, but for illustration purposes.

    def next_label():
        # Increment the label and turn it into a character.
        label = chr(next_label_ord[0])
        next_label_ord[0] += 1
        return label

    nn = JmolNN()
    disorder_groups = []

    for group in unique_disorder_groups:
        # Build filtered lists for each disorder group.
        # This includes the omnipresent (group "." or "0") and this specific group's atoms.
        filtered_coordinates_list = []
        filtered_species_list = []

        for site, site_group in zip(structure, site_groups):
            if site_group in (".", "0") or site_group == group:
                filtered_coordinates_list.append(site.coords)
                filtered_species_list.append(list(site.species.as_dict().keys())[0])

        # Now we find the individual molecules. First create a new structure with only the relevant sites.
        group_structure = Structure(
            lattice = structure.lattice,
            species = filtered_species_list,
            coords  = filtered_coordinates_list,
            coords_are_cartesian = True,
        )
        bond_structure = nn.get_bonded_structure(group_structure)

        # Break this up into molecules.
        # e.g. components[0]["molecule_graph"].molecule.sites contains all of the stuff in the first molecule.
        components = get_structure_components(bond_structure, inc_site_ids=True, inc_molecule_graph=True)

        group_molecules = []

        # Loop over molecules.
        for component in components:
            mol_graph = component["molecule_graph"]

            # Loop over sites in each molecule to extract coordinates and species.
            mol_coordinates = []
            mol_species = []
            sites = mol_graph.molecule.sites

            for site in sites:
                # Get the element. Should only be one in the dict.
                site_dict = site.species.as_dict()
                element = list(site_dict.keys())[0]

                mol_coordinates.append(site.coords)
                mol_species.append(element)

            # Move the molecule to the origin without flattening it.
            mol_coordinates = np.array(mol_coordinates)
            mol_coordinates = mol_coordinates - mol_coordinates.mean(axis=0)

            # Check this molecule against each known conformer using graph matching + Kabsch.
            best_label = None
            best_rmsd = np.inf
            best_R = None

            for label, ref_coords in conformer_coordinates.items():
                mapper = isomorphism.GraphMatcher(conformer_graphs[label], mol_graph.graph.to_undirected(), node_match=node_match)

                if not mapper.is_isomorphic():
                    continue

                # Check each isomorphism, and pick the one with the lowest RMSD against the reference.
                for mapping in mapper.isomorphisms_iter():
                    new_indices = [mapping[ref_idx] for ref_idx in range(len(mapping))]
                    x_prime = mol_coordinates[new_indices].T
                    x_ref = ref_coords.T
                    R = kabsch_rotation(x_ref, x_prime)
                    rmsd = np.sqrt(np.mean(np.sum((x_ref - R.T @ x_prime) ** 2, axis=0)))
                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                        best_R = R
                        best_label = label

            # If the best match is above threshold, this is a new conformer.
            if best_rmsd >= CONFORMER_RMSD_THRESHOLD:
                best_label = next_label()
                conformer_coordinates[best_label] = mol_coordinates
                conformer_species[best_label] = mol_species
                conformer_graphs[best_label] = mol_graph.graph.to_undirected()
                best_R = np.eye(3) # The map is just the identity for the new conformer.

            # Find the proper rotation matrix for this conformer, and save the determinant for future reference.
            det_R = float(np.linalg.det(best_R))
            proper_rotation = det_R * best_R

            group_molecules.append({
                "conformer_label": best_label,
                "det_rotation": det_R,
                "proper_rotation": proper_rotation.tolist(),
            })

        disorder_groups.append({
            "group": "1" if group is None else str(group),
            "occupancy": group_to_occ.get(group, 1.0),
            "molecules": group_molecules,
        })

    conformers = {}
    conformer_labels = list(conformer_coordinates)

    for label in conformer_labels:
        conformers[label] = {
            "atom_symbols": list(conformer_species[label]),
            "atom_coordinates": conformer_coordinates[label].tolist(),
        }

    crystal_data = {
        "cif_path": str(cif_path),
        "conformer_labels": conformer_labels,
        "conformers": conformers,
        "disorder_groups": disorder_groups,
    }

    return crystal_data


def build_crystal_metadata(crystal_data):
    """
    Build the minimal crystal metadata required to reconstruct the crystal later.
    Coordinates will be saved in the td-dft output, so no need to save them here.

    # Arguments:
    - crystal_data::dict: Crystal data returned by get_crystal_data.

    # Returns:
    - crystal_metadata::dict: The reduced crystal metadata for JSON output.
    """

    disorder_groups = []
    for disorder_group in crystal_data["disorder_groups"]:
        group_molecules = []
        
        # Write out the molecules in each disorder group.
        for molecule in disorder_group["molecules"]:
            group_molecules.append({
                "conformer_label": molecule["conformer_label"],
                "det_rotation": molecule["det_rotation"],
                "proper_quaternion": rotation_matrix_to_quaternion(molecule["proper_rotation"]),
            })

        # Append each disorder group.
        disorder_groups.append({
            "group": disorder_group["group"],
            "occupancy": disorder_group["occupancy"],
            "molecules": group_molecules,
        })

    crystal_metadata = {
        "conformer_labels": list(crystal_data["conformer_labels"]),
        "disorder_groups": disorder_groups,
    }

    return crystal_metadata
