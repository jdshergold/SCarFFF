module ReadBasisSet

using HDF5

export get_molecular_data, MoleculeData

# Create a map from real spherical harmonic labels to m.
# For p and d orbitals the labels are named (e.g. "x", "z^2"); for f and above they are
# integer strings matching the m value (e.g. "-3", "0", "3").
const REAL_HARMONIC_MAP = Dict{String, Int}(
    # p-orbitals.
    "y" => -1,
    "z" => 0,
    "x" => 1,
    # d-orbitals.
    "xy" => -2,
    "yz" => -1,
    "z^2" => 0,
    "xz" => 1,
    "x2-y2" => 2,
    # f and above: labels are integer strings matching the m value (e.g. "-3", "0", "3").
    "-6" => -6, "-5" => -5, "-4" => -4, "-3" => -3, "-2" => -2, "-1" => -1,
    "0" => 0,
    "1" => 1, "2" => 2, "3" => 3, "4" => 4, "5" => 5, "6" => 6,
)

# Per-l Cartesian expansion maps.
# Each entry maps a spherical harmonic label to a list of (a, b, c, prefactor) tuples
# representing the Cartesian monomials x^a y^b z^c that sum to that harmonic.
# Note: f and above use integer string labels (e.g. "-3", "0") so each l needs its own map
# to avoid key collisions between orbital types with overlapping m ranges.
const CARTESIAN_EXPANSION_P_MAP = Dict{String, Vector{Tuple{Int, Int, Int, Float64}}}(
    "y" => [(0, 1, 0, 1.0)],
    "z" => [(0, 0, 1, 1.0)],
    "x" => [(1, 0, 0, 1.0)],
)

const CARTESIAN_EXPANSION_D_MAP = Dict{String, Vector{Tuple{Int, Int, Int, Float64}}}(
    "xy"    => [(1, 1, 0, 1.0)],
    "yz"    => [(0, 1, 1, 1.0)],
    "z^2"   => [(0, 0, 2, 2.0), (2, 0, 0, -1.0), (0, 2, 0, -1.0)], # 2z^2 - x^2 - y^2.
    "xz"    => [(1, 0, 1, 1.0)],
    "x2-y2" => [(2, 0, 0, 1.0), (0, 2, 0, -1.0)], # x^2 - y^2.
)

const CARTESIAN_EXPANSION_F_MAP = Dict{String, Vector{Tuple{Int, Int, Int, Float64}}}(
    "-3" => [(2, 1, 0, 3.0), (0, 3, 0, -1.0)],                    # 3x^2 y - y^3.
    "-2" => [(1, 1, 1, 1.0)],                                       # xyz.
    "-1" => [(2, 1, 0, -1.0), (0, 3, 0, -1.0), (0, 1, 2, 4.0)],   # y(5z^2 - r^2).
    "0"  => [(2, 0, 1, -3.0), (0, 2, 1, -3.0), (0, 0, 3, 2.0)],   # 5z^3 - 3zr^2.
    "1"  => [(3, 0, 0, -1.0), (1, 2, 0, -1.0), (1, 0, 2, 4.0)],   # x(5z^2 - r^2).
    "2"  => [(2, 0, 1, 1.0), (0, 2, 1, -1.0)],                     # z(x^2 - y^2).
    "3"  => [(3, 0, 0, 1.0), (1, 2, 0, -3.0)],                     # x(x^2 - 3y^2).
)

const CARTESIAN_EXPANSION_G_MAP = Dict{String, Vector{Tuple{Int, Int, Int, Float64}}}(
    "-4" => [(3, 1, 0, 1.0), (1, 3, 0, -1.0)],
    "-3" => [(2, 1, 1, 3.0), (0, 3, 1, -1.0)],
    "-2" => [(3, 1, 0, -1.0), (1, 3, 0, -1.0), (1, 1, 2, 6.0)],
    "-1" => [(2, 1, 1, -3.0), (0, 3, 1, -3.0), (0, 1, 3, 4.0)],
    "0"  => [(4, 0, 0, 3.0), (2, 2, 0, 6.0), (0, 4, 0, 3.0), (2, 0, 2, -24.0), (0, 2, 2, -24.0), (0, 0, 4, 8.0)],
    "1"  => [(3, 0, 1, -3.0), (1, 2, 1, -3.0), (1, 0, 3, 4.0)],
    "2"  => [(4, 0, 0, -1.0), (0, 4, 0, 1.0), (2, 0, 2, 6.0), (0, 2, 2, -6.0)],
    "3"  => [(3, 0, 1, 1.0), (1, 2, 1, -3.0)],
    "4"  => [(4, 0, 0, 1.0), (2, 2, 0, -6.0), (0, 4, 0, 1.0)],
)

const CARTESIAN_EXPANSION_H_MAP = Dict{String, Vector{Tuple{Int, Int, Int, Float64}}}(
    "-5" => [(4, 1, 0, 5.0), (2, 3, 0, -10.0), (0, 5, 0, 1.0)],
    "-4" => [(3, 1, 1, 1.0), (1, 3, 1, -1.0)],
    "-3" => [(4, 1, 0, -3.0), (2, 3, 0, -2.0), (0, 5, 0, 1.0), (2, 1, 2, 24.0), (0, 3, 2, -8.0)],
    "-2" => [(3, 1, 1, -1.0), (1, 3, 1, -1.0), (1, 1, 3, 2.0)],
    "-1" => [(4, 1, 0, 1.0), (2, 3, 0, 2.0), (0, 5, 0, 1.0), (2, 1, 2, -12.0), (0, 3, 2, -12.0), (0, 1, 4, 8.0)],
    "0"  => [(4, 0, 1, 15.0), (2, 2, 1, 30.0), (0, 4, 1, 15.0), (2, 0, 3, -40.0), (0, 2, 3, -40.0), (0, 0, 5, 8.0)],
    "1"  => [(5, 0, 0, 1.0), (3, 2, 0, 2.0), (1, 4, 0, 1.0), (3, 0, 2, -12.0), (1, 2, 2, -12.0), (1, 0, 4, 8.0)],
    "2"  => [(4, 0, 1, -1.0), (0, 4, 1, 1.0), (2, 0, 3, 2.0), (0, 2, 3, -2.0)],
    "3"  => [(5, 0, 0, -1.0), (3, 2, 0, 2.0), (1, 4, 0, 3.0), (3, 0, 2, 8.0), (1, 2, 2, -24.0)],
    "4"  => [(4, 0, 1, 1.0), (2, 2, 1, -6.0), (0, 4, 1, 1.0)],
    "5"  => [(5, 0, 0, 1.0), (3, 2, 0, -10.0), (1, 4, 0, 5.0)],
)

const CARTESIAN_EXPANSION_I_MAP = Dict{String, Vector{Tuple{Int, Int, Int, Float64}}}(
    "-6" => [(5, 1, 0, 3.0), (3, 3, 0, -10.0), (1, 5, 0, 3.0)],
    "-5" => [(4, 1, 1, 5.0), (2, 3, 1, -10.0), (0, 5, 1, 1.0)],
    "-4" => [(5, 1, 0, 1.0), (1, 5, 0, -1.0), (3, 1, 2, -10.0), (1, 3, 2, 10.0)],
    "-3" => [(4, 1, 1, -9.0), (2, 3, 1, -6.0), (0, 5, 1, 3.0), (2, 1, 3, 24.0), (0, 3, 3, -8.0)],
    "-2" => [(5, 1, 0, 1.0), (3, 3, 0, 2.0), (1, 5, 0, 1.0), (3, 1, 2, -16.0), (1, 3, 2, -16.0), (1, 1, 4, 16.0)],
    "-1" => [(4, 1, 1, 5.0), (2, 3, 1, 10.0), (0, 5, 1, 5.0), (2, 1, 3, -20.0), (0, 3, 3, -20.0), (0, 1, 5, 8.0)],
    "0"  => [(6, 0, 0, 5.0), (4, 2, 0, 15.0), (2, 4, 0, 15.0), (0, 6, 0, 5.0), (4, 0, 2, -90.0), (2, 2, 2, -180.0), (0, 4, 2, -90.0), (2, 0, 4, 120.0), (0, 2, 4, 120.0), (0, 0, 6, -16.0)],
    "1"  => [(5, 0, 1, 5.0), (3, 2, 1, 10.0), (1, 4, 1, 5.0), (3, 0, 3, -20.0), (1, 2, 3, -20.0), (1, 0, 5, 8.0)],
    "2"  => [(6, 0, 0, 1.0), (4, 2, 0, 1.0), (2, 4, 0, -1.0), (0, 6, 0, -1.0), (4, 0, 2, -16.0), (0, 4, 2, 16.0), (2, 0, 4, 16.0), (0, 2, 4, -16.0)],
    "3"  => [(5, 0, 1, -3.0), (3, 2, 1, 6.0), (1, 4, 1, 9.0), (3, 0, 3, 8.0), (1, 2, 3, -24.0)],
    "4"  => [(6, 0, 0, -1.0), (4, 2, 0, 5.0), (2, 4, 0, 5.0), (0, 6, 0, -1.0), (4, 0, 2, 10.0), (2, 2, 2, -60.0), (0, 4, 2, 10.0)],
    "5"  => [(5, 0, 1, 1.0), (3, 2, 1, -10.0), (1, 4, 1, 5.0)],
    "6"  => [(6, 0, 0, 1.0), (4, 2, 0, -15.0), (2, 4, 0, 15.0), (0, 6, 0, -1.0)],
)

@inline function cartesian_map_for_l(l::Int)
    l == 1 && return CARTESIAN_EXPANSION_P_MAP
    l == 2 && return CARTESIAN_EXPANSION_D_MAP
    l == 3 && return CARTESIAN_EXPANSION_F_MAP
    l == 4 && return CARTESIAN_EXPANSION_G_MAP
    l == 5 && return CARTESIAN_EXPANSION_H_MAP
    l == 6 && return CARTESIAN_EXPANSION_I_MAP
    error("No Cartesian expansion map for l=$l")
end

# Create a map from angular momentum quantum number labels to their corresponding integer values.
const L_MAP = Dict{Char, Int}(
    'S' => 0,
    'P' => 1,
    'D' => 2,
    'F' => 3,
    'G' => 4,
    'H' => 5,
    'I' => 6,
)

const AU_TO_ANGSTROM = 0.529177 # For converting Bohr radii to Angstroms.

@inline function canonical_basis_name(name::AbstractString)
    return replace(lowercase(strip(name)), "*" => "s", r"[-_\s\(\)]" => "")
end

function resolve_basis_set_path(basis_set_name::AbstractString)
    basis_dir = joinpath(@__DIR__, "..", "data", "basis_sets")
    basis_filename = "$(canonical_basis_name(basis_set_name)).h5"
    basis_path = joinpath(basis_dir, basis_filename)
    isfile(basis_path) && return basis_path

    error("Precomputed basis set file not found for basis '$basis_set_name' at $basis_path")
end

@inline function parse_ao_label(descriptor::AbstractString)
    """
    Parse the atomic-orbital string from a PySCF AO label.

    Supports descriptors like "1s", "2px", "5g-4", "5g+4", "10s", and "10g+4".

    # Returns
    - n: Principal quantum number.
    - l: Orbital angular momentum quantum number.
    - harmonic_label: Harmonic suffix with a leading `+` stripped when present.
    - m: Magnetic quantum number for the real spherical harmonic.
    """
    match_result = match(r"^(\d+)([A-Za-z])(.*)$", descriptor)
    match_result === nothing && error("Unsupported AO label '$descriptor'.")

    n = parse(Int, match_result.captures[1])
    l_char = uppercase(only(match_result.captures[2]))
    haskey(L_MAP, l_char) || error("Unsupported AO angular momentum '$l_char' in label '$descriptor'.")
    l = L_MAP[l_char]

    harmonic_label = lstrip(match_result.captures[3], '+')
    m = isempty(harmonic_label) ? 0 : get(REAL_HARMONIC_MAP, harmonic_label) do
        error("Unsupported AO harmonic label '$harmonic_label' in AO label '$descriptor'.")
    end

    return (n = n, l = l, harmonic_label = harmonic_label, m = m)
end

@inline function restore_python_hdf5_order(data::AbstractArray)
    """
    Reverse the dimension order of data written by Python/HDF5 so that it matches Julia's column-major layout.

    HDF5.jl reads datasets written by h5py with dimensions reversed (e.g. (N, 3) becomes (3, N)),
    so mirror the fix used in scripts/plot_spherical_slices.py by flipping the axes back.
    """
    nd = ndims(data)
    nd <= 1 && return data
    return permutedims(data, nd:-1:1)
end

struct MoleculeData{T<:AbstractFloat}
    """
    Efficient data structure for molecular orbital calculations.

    A flattened array representation of molecular orbital datahat is optimised for vectorised operations
    while maintaining hierarchicalrelationships between atoms, orbitals, primitives, and Cartesian terms. 
    Also stores transition matrices for excited states.

    All primitives and coordinates here are in units of Angstrom to some appropriate power.
    """
    # Primitive data.
    widths::Vector{T} # Gaussian widths.
    coefficients::Vector{T} # Raw contraction coefficients d_μ.
    normalised_coefficients::Vector{T} # Fully normalised primitive prefactors d_μ ξ_μ N_α.

    # Atom coordinates and edges of a cuboid about the molecule.
    atom_coordinates::Array{T, 2} # Coordinates of the atoms in the molecule.
    molecule_cuboid::Array{T, 2} # Cuboid edges for the entire molecule (min and max coordinates).

    # Index mappings for hierarchy.
    primitive_to_orbital::Vector{Int} # Maps primitive index to orbital index.
    orbital_to_atom::Vector{Int} # Maps orbital index to atom index.
    primitive_to_atom::Vector{Int} # Maps primitive index to atom index. This is just to save computing orbital_to_atom[primitive_to_orbital[i]] repeatedly.
    cartesian_term_to_primitive::Vector{Int} # Maps each Cartesian term to its primitive index.
    cartesian_term_to_orbital::Vector{Int} # Maps each Cartesian term to its orbital index.
    cartesian_term_to_atom::Vector{Int} # Maps each Cartesian term to its atom index.

    orbital_n::Vector{Int} # Principal quantum number of each orbital.
    orbital_l::Vector{Int} # Orbital angular momentum of each orbital.
    orbital_m::Vector{Int} # Magnetic quantum number of each orbital.

    # Cartesian angular momentum expansion.
    # Each primitive expands into multiple Cartesian terms (e.g., z^2 -> 2z^2 - x^2 - y^2).
    cartesian_a::Vector{Int} # Power of x in each Cartesian term.
    cartesian_b::Vector{Int} # Power of y in each Cartesian term.
    cartesian_c::Vector{Int} # Power of z in each Cartesian term.
    cartesian_prefactor::Vector{T} # Prefactor for each Cartesian term (e.g., 2.0, -1.0).


    # Transition matrices for excited states.
    transition_matrices::Vector{Array{T, 2}}

    # Transition energies in eV.
    transition_energies_eV::Vector{T}

    # Dimensions.
    n_primitives::Int
    n_orbitals::Int
    n_atoms::Int
    n_cartesian_terms::Int
end

function construct_molecular_data(h5_data::Dict, basis_h5_path::String; precision::Type{T} = Float64)::MoleculeData{T} where {T<:AbstractFloat}
    """
    Construct a MoleculeData structure from the precomputed basis set HDF5 file and the TD-DFT results, according to the order of the atomic orbitals listed there.

    # Arguments:
    - h5_data::Dict: The parsed TD-DFT HDF5 data containing atom coordinates and orbital ordering.
    - basis_h5_path::String: Path to the precomputed basis set HDF5 file.

    # Returns:
    - MoleculeData: A structure containing the molecular orbitals data.
    """

    # Initialise arrays for the molecular orbitals.
    widths = T[]
    coefficients = T[]
    normalised_coefficients = T[]

    # Convert atom coordinates from Bohr to Angstrom.
    atom_coordinates = h5_data["atom_coordinates"] .* T(AU_TO_ANGSTROM)
    molecule_cuboid = Array{T}(undef, 0, 3)
    primitive_to_orbital = Int[]
    orbital_to_atom = Int[]
    primitive_to_atom = Int[]
    orbital_n = Int[]
    orbital_l = Int[]
    orbital_m = Int[]
    cartesian_a = Int[]
    cartesian_b = Int[]
    cartesian_c = Int[]
    cartesian_prefactor = T[]
    cartesian_term_to_primitive = Int[]
    cartesian_term_to_orbital = Int[]
    cartesian_term_to_atom = Int[]
    transition_matrices = Vector{Array{T, 2}}(undef, 0)
    transition_energies_eV = T[]

    # Now that we have the atom coordinates, we can determine the number of atoms and cuboid.
    n_atoms = size(atom_coordinates, 1)
    molecule_cuboid = vcat(minimum(atom_coordinates, dims = 1), maximum(atom_coordinates, dims = 1)) # Minimum and maximum x, y, z coordinates.

    # Now read in the transition matrices and energies.
    n_transitions = length(h5_data["transition_matrices"])
    for idx in 1:n_transitions
        push!(transition_matrices, h5_data["transition_matrices"][idx])
        push!(transition_energies_eV, h5_data["energies_ev"][idx])
    end

    # Open the basis set HDF5 file once and read the data for all orbitals.
    h5open(basis_h5_path, "r") do basis_h5
        # Now parse each orbital and fill the remaining arrays.
        for (orbital_idx, label) in enumerate(h5_data["ao_labels"])
            label = split(strip(label))
            atom_idx = parse(Int, label[1]) + 1 # Atom index, 1-based in Julia, hence we add the 1.
            species = label[2] # Element symbol, e.g. "C", "O".
            parsed_descriptor = parse_ao_label(label[3])
            n = parsed_descriptor.n
            l = parsed_descriptor.l
            harmonic_label = parsed_descriptor.harmonic_label
            m = parsed_descriptor.m

            # Now push to the arrays.
            push!(orbital_to_atom, atom_idx)
            push!(orbital_n, n)
            push!(orbital_l, l)
            push!(orbital_m, m)

            # Get Cartesian expansion terms for this orbital.
            if l == 0
                # s-orbitals are just (0,0,0) with prefactor 1.
                harmonic_label = ""
                cartesian_terms = [(0, 0, 0, 1.0)]
            else
                cartesian_terms = cartesian_map_for_l(l)[harmonic_label]
            end

            # Read the corresponding primitive data from the basis HDF5 file.
            group_name = "$(species)_$(n)_$(l)"

            # Read widths and coefficients for this (species, n, l) combination.
            orbital_widths = read(basis_h5[group_name], "widths")
            orbital_coeffs = read(basis_h5[group_name], "coefficients")

            # Determine the dataset name for normalised coefficients.
            if l == 0
                # s-orbitals use the label "normalised_coefficients".
                dataset_name = "normalised_coefficients"
            else
                # The other orbitals use the name "norm_$(sanitised_label)".
                sanitised_label = replace(replace(harmonic_label, "^" => ""), "-" => "_")
                dataset_name = "norm_$(sanitised_label)"
            end
            orbital_normalised_coeffs = read(basis_h5[group_name], dataset_name)

            n_primitives_for_orbital = length(orbital_widths)

            # Loop over primitives within this orbital.
            @inbounds for i in 1:n_primitives_for_orbital
                width = T(orbital_widths[i])
                coeff = T(orbital_coeffs[i])
                normalised_coeff = T(orbital_normalised_coeffs[i])

                push!(widths, width)
                push!(coefficients, coeff)                         # d_μ.
                push!(normalised_coefficients, normalised_coeff)  # d_μ ξ_μ N_α.
                push!(primitive_to_orbital, orbital_idx)
                push!(primitive_to_atom, atom_idx)

                current_primitive_idx = length(widths) # Get the current primitive idx, that will be shared by all Cartesian terms for that orbital.
                for (a, b, c, prefactor) in cartesian_terms
                    push!(cartesian_a, a)
                    push!(cartesian_b, b)
                    push!(cartesian_c, c)
                    push!(cartesian_prefactor, T(prefactor))
                    push!(cartesian_term_to_primitive, current_primitive_idx)
                    push!(cartesian_term_to_orbital, orbital_idx)
                    push!(cartesian_term_to_atom, atom_idx)
                end
            end
        end
    end
    # Now we have the total number of primitives, orbitals, and Cartesian terms.
    n_primitives = length(widths)
    n_orbitals = length(orbital_to_atom)
    n_cartesian_terms = length(cartesian_a)

    # Create the MoleculeData structure.
    return MoleculeData{T}(
        widths,
        coefficients,
        normalised_coefficients,
        atom_coordinates,
        molecule_cuboid,
        primitive_to_orbital,
        orbital_to_atom,
        primitive_to_atom,
        cartesian_term_to_primitive,
        cartesian_term_to_orbital,
        cartesian_term_to_atom,
        orbital_n,
        orbital_l,
        orbital_m,
        cartesian_a,
        cartesian_b,
        cartesian_c,
        cartesian_prefactor,
        transition_matrices,
        transition_energies_eV,
        n_primitives,
        n_orbitals,
        n_atoms,
        n_cartesian_terms
    )
end

function get_molecular_data(
        td_dft_file::String;
        precision::Type{T} = Float64,
    )::MoleculeData{T} where {T<:AbstractFloat}
    """
    Convenience function to read in the basis set and TD-DFT data, and construct the MoleculeData structure.

    # Arguments:
    - td_dft_file::String: Path to the TD-DFT HDF5 file containing atom coordinates and orbital ordering.
    - precision::Type{T}: The floating point precision to use. Defaults to Float64.

    # Returns:
    - molecule_data: A structure containing the molecular orbitals data.
    """
    # Load the TD-DFT HDF5 file.
    h5_data = h5open(td_dft_file, "r") do io
        # Read the metadata (basis set and SMILES) from attributes.
        basis_set_name = read_attribute(io, "basis")

        # Read the datasets.
        atom_symbols = read(io, "atom_symbols")
        atom_coordinates = restore_python_hdf5_order(read(io, "atom_coordinates"))
        ao_labels = read(io, "ao_labels")
        energies_ev = read(io, "energies_ev")

        # Read the transition matrices, which are stored under different keys.
        transition_matrices = []
        state_idx = 1
        while haskey(io, "d_ij_state_$state_idx")
            raw_tdm = read(io, "d_ij_state_$state_idx")
            push!(transition_matrices, restore_python_hdf5_order(raw_tdm))
            state_idx += 1
        end

        Dict(
            "basis" => basis_set_name,
            "atom_symbols" => atom_symbols,
            "atom_coordinates" => atom_coordinates,
            "ao_labels" => ao_labels,
            "energies_ev" => energies_ev,
            "transition_matrices" => transition_matrices
        )
    end

    # Construct the path to the precomputed basis set HDF5 file.
    basis_set_name = h5_data["basis"]
    basis_set_path = resolve_basis_set_path(basis_set_name)

    # Construct the molecular data using the TD-DFT HDF5 data and precomputed basis set.
    molecule_data = construct_molecular_data(h5_data, basis_set_path; precision=precision)

    return molecule_data
end

end
