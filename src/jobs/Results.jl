"""Write aggregate molecular results, with shared grids and one group per transition."""
module MolecularResults

using HDF5

export write_results

function set_attribute(node, name, value)
    """
    Write an attribute, replacing one of the same name left by an earlier conformer.

    # Arguments:
    - node: The HDF5 file or group.
    - name::String: The attribute name.
    - value: The attribute value.
    """
    haskey(attributes(node), name) && delete_attribute(node, name)
    attributes(node)[name] = value
end

function write_quantity(group, name, value, units=nothing)
    """
    Write a physical quantity and its dimensional units.

    # Arguments:
    - group: The HDF5 group.
    - name::String: The dataset name.
    - value: The quantity to save.
    - units: Its units, or nothing for a dimensionless quantity.
    """
    write(group, name, value)
    if units !== nothing
        attributes(group[name])["units"] = units
    end
end

function write_results(path, mode, signature, grid, indices, energies;
                       f_s=nothing, f_sq_lm=nothing, R_tensor=nothing,
                       transition_density=nothing, V_tensors=nothing,
                       check_result=nothing, group=nothing)
    """
    Save one complete molecular calculation before replacing its previous result file.

    # Arguments:
    - path::String: The result filename.
    - mode::String: The form factor method.
    - signature::String: The calculation signature.
    - grid::Dict: Shared coordinates and their units.
    - indices::Vector{Int}: The original transition indices.
    - energies::Vector: The transition energies, in eV.
    - f_s: Complex form factors, with transition index first.
    - f_sq_lm: Real intensity coefficients, with transition index first.
    - R_tensor: The R tensors, with transition index first.
    - transition_density: The Fourier transition densities, in Angstrom^-3.
    - V_tensors: Shared Cartesian V_x, V_y and V_z tensors.
    - check_result: An optional check of the completed temporary file.
    - group: The group to save the transitions under, which is "conformers/A" and so on for a
      crystal, whose conformers share one file. Nothing puts them at the file's root.
    """
    # A single molecule is written to a temporary file and moved into place once it is whole. A
    # crystal's conformers share one file, so each is added to the file that is already there.
    temporary_path = path * ".tmp"
    target, mode_flag = group === nothing ? (temporary_path, "w") : (path, "cw")
    try
        h5open(target, mode_flag) do io
            # Tag the file with the method it came from and the settings that produced it.
            set_attribute(io, "mode", mode)
            set_attribute(io, "stage_signature", signature)

            # Save the coordinate grids, which every transition shares, and so sit at the root
            # whether or not the transitions themselves are in conformer groups.
            if !haskey(io, "grid")
                coordinates = create_group(io, "grid")
                for (name, (values, units)) in grid
                    write_quantity(coordinates, name, values, units)
                end
            end

            # Start from nothing if this group was written by an earlier run.
            transitions_path = group === nothing ? "transitions" : "$(group)/transitions"
            haskey(io, transitions_path) && delete_object(io, transitions_path)

            # One group per transition, holding its energy and whichever quantities were asked for.
            transitions = create_group(io, transitions_path)
            for (i, s) in enumerate(indices)
                state = create_group(transitions, string(s))
                write_quantity(state, "Delta_E", energies[i], "eV")
                for (name, values, units) in (("f_s", f_s, nothing),
                        ("f_sq_lm", f_sq_lm, nothing), ("R_tensor", R_tensor, nothing),
                        ("transition_density", transition_density, "angstrom^-3"))
                    if values !== nothing
                        write_quantity(state, name, selectdim(values, 1, i) |> Array, units)
                    end
                end
            end
            # The Cartesian V tensors sit outside the transitions, since all of them share one set.
            if V_tensors !== nothing
                tensors = create_group(io, "cartesian_tensors")
                for (name, values) in zip(("V_x", "V_y", "V_z"), V_tensors)
                    write(tensors, name, values)
                end
            end

            # Mark the file finished, so a truncated one can be told apart from a complete one.
            set_attribute(io, "completed", true)
        end

        if group === nothing
            # Check the finished file before it replaces the previous result.
            if check_result !== nothing
                check_result(temporary_path)
            end
            mv(temporary_path, path; force=true)
        end
    catch
        # Leave no half-written file behind.
        group === nothing && rm(temporary_path; force=true)
        rethrow()
    end
end

end
