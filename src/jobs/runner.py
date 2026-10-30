"""Run explicit TD-DFT and form factor stages from a job YAML."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import h5py
import numpy as np
import yaml

from .config import ROOT, load_job, read_systems
from ..results import text_attribute
from .output import print_heading, print_stage_timings


def content_hash(path):
    """
    Identify the contents of a saved calculation or input file.

    # Arguments:
    - path::Path: The file.

    # Returns:
    - str: The hash of the file contents.
    """
    # Read in blocks, so a large saved calculation is never held in memory all at once.
    contents = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(2**20), b''):
            contents.update(block)
    return contents.hexdigest()


def tddft_paths(job, directory):
    """
    Find the TD-DFT files belonging to one molecule or crystal.

    # Arguments:
    - job::dict: The resolved job.
    - directory::Path: The numbered result directory.

    # Returns:
    - list[Path]: The molecular or conformer files.
    """
    if job['mode'] != 'crystal':
        return [directory / 'td_dft.h5']
    with (directory / 'crystal_metadata.json').open() as source:
        labels = json.load(source)['conformer_labels']
    if not labels:
        raise ValueError('The crystal has no conformers.')
    suffix = 'f32' if job['execution']['precision'] == 'float32' else 'f64'
    return [directory / 'conformers' / label / f'td_dft_results_{suffix}.h5' for label in labels]


def calculation_signature(job, system, stage, directory):
    """
    Identify the physical inputs and settings for one calculation stage.

    # Arguments:
    - job::dict: The resolved job.
    - system::dict: The molecule or crystal input.
    - stage::str: tddft or form_factors.
    - directory::Path: The numbered result directory.

    # Returns:
    - str: The calculation signature.
    """
    precision = job['execution']['precision']

    # TD-DFT depends on the molecule and the level of theory. A crystal is identified by its CIF
    # contents rather than the file it was read from, so editing the structure invalidates the old
    # result but moving the file does not. A molecule is identified by its SMILES.
    if stage == 'tddft':
        parameters = {'nstates': job['nstates'], 'tddft': job['tddft'], 'precision': precision}
        if job['mode'] == 'crystal':
            parameters['cif_contents'] = content_hash(system['input'])
        else:
            parameters['input'] = system['input']
    # The form factors depend on the grids and the TD-DFT they were built from.
    else:
        parameters = {key: job[key] for key in ['mode', 'transitions', 'verification']}
        parameters['precision'] = precision
        for section in ['spherical', 'fourier', 'cartesian', 'crystal']:
            if section in job:
                parameters[section] = job[section]
        parameters['tddft_contents'] = [content_hash(path) for path in tddft_paths(job, directory)]
    return hashlib.sha256(json.dumps(parameters, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_tddft(path, nstates, signature, precision):
    """
    Require the requested saved states and their three-state computed buffer.

    # Arguments:
    - path::Path: The TD-DFT file.
    - nstates::int: The number of saved states.
    - signature::str: The expected calculation signature.
    - precision::str: The saved precision.
    """
    with h5py.File(path) as results:
        # The settings must match, or the saved file belongs to a different calculation.
        if text_attribute(results, 'stage_signature') != signature:
            raise ValueError(f'{path}: TD-DFT settings or input do not match; set tddft to run.')

        # The energies and oscillator strengths, one of each per saved state.
        if results['Delta_E_eV'].shape != (nstates,) or results['f_osc'].shape != (nstates,):
            raise ValueError(f'{path}: expected {nstates} saved states.')

        # Three states beyond the saved ones are computed, so a degenerate pair is never split.
        if results['Delta_E_eV'].dtype.name != precision or results.attrs.get('nstates_td_dft', 0) < nstates + 3:
            raise ValueError(f'{path}: precision or computed state count does not match.')

        # The X and Y amplitudes, which the densities are built from.
        expected = {f'{amplitude}_state_{s}' for amplitude in ('X', 'Y') for s in range(1, nstates + 1)}
        saved = {name for name in results if name.startswith(('X_state_', 'Y_state_'))}
        if saved != expected:
            raise ValueError(f'{path}: saved X and Y amplitudes do not match nstates.')


def form_factor_path(job, directory):
    """
    Locate the aggregate molecular or existing crystal result file.

    # Arguments:
    - job::dict: The resolved job.
    - directory::Path: The numbered result directory.

    # Returns:
    - Path: The form factor file.
    """
    if job['mode'] != 'crystal':
        return directory / 'form_factors.h5'
    suffix = 'f32' if job['execution']['precision'] == 'float32' else 'f64'
    return directory / 'crystal' / f'crystal_form_factors_{suffix}.h5'


def validate_form_factors(job, path, signature):
    """
    Require completed form factors with matching calculation settings.

    # Arguments:
    - job::dict: The resolved job.
    - path::Path: The saved form factor file.
    - signature::str: The expected calculation signature.
    """
    with h5py.File(path) as results:
        # The settings must match, and the file must have been finished rather than truncated.
        if text_attribute(results, 'stage_signature') != signature or not results.attrs.get('completed', False):
            raise ValueError(f'{path}: form factors do not match; set form_factors to run.')

        # Crystals keep their own layout, with a file per conformer alongside the crystal result.
        if job['mode'] == 'crystal':
            # The incoherent reference is only written for a coherent crystal, because an unmixed
            # one already is the reference.
            required = ['f_sq_lm']
            if job['crystal']['structure_function']:
                required.append('S_lm')
            if job['crystal']['incoherent_reference'] and job['crystal']['order'] == 'coherent':
                required.append('f_sq_lm_incoherent')
            if any(name not in results for name in required):
                raise ValueError(f'{path}: requested crystal quantities are missing.')
            indices = range(1, job['nstates'] + 1) if job['transitions'] == 'all' else job['transitions']
            suffix = 'f32' if job['execution']['precision'] == 'float32' else 'f64'
            for td_path in tddft_paths(job, path.parent.parent):
                for s in indices:
                    conformer_path = td_path.parent / str(s) / f'fs_grid_{suffix}.h5'
                    with h5py.File(conformer_path) as conformer:
                        if any(name not in conformer for name in job['spherical']['save']):
                            raise ValueError(f'{conformer_path}: requested conformer quantities are missing.')
            return
        # The saved transitions must be the ones asked for, by the method asked for.
        indices = list(range(1, job['nstates'] + 1)) if job['transitions'] == 'all' else job['transitions']
        if text_attribute(results, 'mode') != job['mode'] or set(results['transitions']) != set(map(str, indices)):
            raise ValueError(f'{path}: method or transition selection does not match.')

        # The coordinate grids, and their units and precision.
        grid_names = ['q', 'theta', 'phi'] if job['mode'] == 'spherical' else ['q_x', 'q_y', 'q_z']
        if any(name not in results['grid'] for name in grid_names):
            raise ValueError(f'{path}: shared coordinates are missing.')
        precision = np.dtype(job['execution']['precision'])
        for name in grid_names:
            coordinate = results[f'grid/{name}']
            if coordinate.ndim != 1 or coordinate.dtype != precision or len(coordinate) < 2:
                raise ValueError(f'{path}: {name} has incorrect precision or shape.')
            if text_attribute(coordinate, 'units') != ('rad' if name in ('theta', 'phi') else 'keV'):
                raise ValueError(f'{path}: {name} units do not match.')
        # Then each transition, which must carry its energy and every quantity that was requested.
        # HDF5 reverses the axis order, so the expected f_s shape is the grid lengths backwards.
        amplitude_shape = tuple(len(results[f'grid/{name}']) for name in reversed(grid_names))
        requested = job.get(job['mode'], {}).get('save', ['f_s'])
        for s in indices:
            group = results[f'transitions/{s}']
            if 'Delta_E' not in group or any(name not in group for name in requested if name != 'V'):
                raise ValueError(f'{path}: transition {s} is incomplete.')
            if group['Delta_E'].shape != () or group['Delta_E'].dtype != precision or text_attribute(group['Delta_E'], 'units') != 'eV':
                raise ValueError(f'{path}: transition {s} has incorrect energy units, precision or shape.')
            for name in requested:
                if name == 'V':
                    continue
                # f_s and the R tensor are complex, so compare against the real part's precision.
                quantity = group[name]
                real_precision = np.empty((), dtype=quantity.dtype).real.dtype
                if real_precision != precision or (name in ('f_s', 'R_tensor') and quantity.dtype.kind != 'c'):
                    raise ValueError(f'{path}: transition {s}, {name} has incorrect precision.')
                if name == 'f_s' and quantity.shape != amplitude_shape:
                    raise ValueError(f'{path}: transition {s}, f_s does not match the shared grid.')
                if name in ('f_sq_lm', 'R_tensor') and (quantity.ndim != 2 or quantity.shape[-1] != len(results['grid/q'])):
                    raise ValueError(f'{path}: transition {s}, {name} does not match the shared radial grid.')
        if 'V' in requested and 'cartesian_tensors' not in results:
            raise ValueError(f'{path}: shared Cartesian tensors are missing.')
        if job['mode'] == 'fourier' and any('transition_density' not in results[f'transitions/{s}'] for s in indices):
            raise ValueError(f'{path}: Fourier transition densities are missing.')


def compute_tddft(job, system, directory):
    """
    Compute one molecule or the conformers of one crystal.

    # Arguments:
    - job::dict: The resolved job.
    - system::dict: The molecule or crystal input.
    - directory::Path: The numbered output directory.
    """
    from .tddft import run_td_dft_analysis, run_td_dft_analysis_from_coordinates
    if job['execution']['threads'] != 'auto':
        from pyscf import lib
        lib.num_threads(job['execution']['threads'])

    # The settings both paths share.
    settings = job['tddft']
    common = dict(basis=settings['basis'], xc=settings['xc'], nstates=job['nstates'],
                  use_gpu=job['execution']['gpu'], precision=job['execution']['precision'])

    # A molecule is built from its SMILES string.
    if job['mode'] != 'crystal':
        run_td_dft_analysis(smiles=system['input'], output_directory=str(directory),
                            ring_flatten=settings['ring_flatten'],
                            dft_optimisation=settings['dft_optimisation'],
                            plot_molecule_3d=settings['plot_molecule_3d'], **common)
        return

    # A crystal is read from its CIF, then run once per distinct conformer in the cell.
    from .crystal import get_crystal_data, build_crystal_metadata, build_coordinate_list
    data = get_crystal_data(system['input'])
    (directory / 'crystal_metadata.json').write_text(json.dumps(build_crystal_metadata(data), indent=2))
    suffix = 'f32' if job['execution']['precision'] == 'float32' else 'f64'
    for label in data['conformer_labels']:
        conformer = data['conformers'][label]
        coordinates = build_coordinate_list(conformer['atom_symbols'], conformer['atom_coordinates'])
        output = directory / 'conformers' / label
        output.mkdir(parents=True, exist_ok=True)
        run_td_dft_analysis_from_coordinates(coordinates, species=f'{system["label"]}_{label}',
                                             output_directory=str(output),
                                             filename=f'td_dft_results_{suffix}.h5', **common)


def run_job(job):
    """
    Run the requested stages, reporting failures without starting unrequested calculations.

    # Arguments:
    - job::dict: The resolved job.
    """
    started = perf_counter()
    run_times = []
    # Save the resolved job and a summary table beside the results, so a run can be identified later.
    systems = read_systems(job)
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'job.yaml').write_text(yaml.safe_dump(job, sort_keys=False))
    with (output / 'metadata.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['folder', 'smiles', 'basis', 'xc'])
        for system in systems:
            writer.writerow([system['number'], system['label'], job['tddft']['basis'], job['tddft']['xc']])

    # Work through the systems, collecting the ones that still need form factors computing.
    work, failures = [], []
    for system in systems:
        directory = output / str(system['number'])
        directory.mkdir(exist_ok=True)
        print_heading('TD-DFT', system['label'], system['number'], len(systems))
        tddft_complete = False
        try:
            # TD-DFT first, stamping the file with its signature so it can be reused later.
            stage = job['stages']['tddft']
            if stage != 'run':
                print(f'  TD-DFT: {"reuse saved results" if stage == "reuse" else "skipped"}', flush=True)
            signature = calculation_signature(job, system, 'tddft', directory)
            if stage == 'run':
                compute_tddft(job, system, directory)
                for path in tddft_paths(job, directory):
                    with h5py.File(path, 'r+') as results:
                        results.attrs['stage_signature'] = signature
                (directory / '.tddft_failed').unlink(missing_ok=True)
            if stage != 'skip':
                for path in tddft_paths(job, directory):
                    validate_tddft(path, job['nstates'], signature, job['execution']['precision'])
                (directory / '.tddft_failed').unlink(missing_ok=True)
            tddft_complete = True

            # Then the form factors, which either reuse a saved result or go on the work list.
            stage = job['stages']['form_factors']
            if stage == 'skip':
                print('  Form factors: skipped', flush=True)
                continue
            signature = calculation_signature(job, system, 'form_factors', directory)
            if stage == 'reuse':
                validate_form_factors(job, form_factor_path(job, directory), signature)
                print('  Form factors: reused saved results', flush=True)
            else:
                work.append({**system, 'form_factor_signature': signature})
        except Exception as error:
            # Carry on to the next system, leaving a marker if its TD-DFT is the part that broke.
            failures.append(f'{system["number"]}: {error}')
            print(f'  Failed: {error}', flush=True)
            if job['stages']['tddft'] == 'run' and not tddft_complete:
                (directory / '.tddft_failed').write_text(str(error) + '\n')

    # Hand the whole work list to Julia in one go, as JSON on its standard input.
    run_times.append(('Preparation and TD-DFT', perf_counter() - started))
    if work:
        settings = {**job, 'systems': work}
        command = ['julia', f'--project={ROOT}', '-t', str(job['execution']['threads']),
                   str(ROOT / 'src/jobs/compute.jl')]
        form_factors_started = perf_counter()
        result = subprocess.run(command, input=json.dumps(settings, allow_nan=False), text=True)
        run_times.append(('Form factors (including Julia startup)', perf_counter() - form_factors_started))

        # Check what Julia wrote, so a silent or partial failure still gets reported.
        for system in work:
            directory = output / str(system['number'])
            try:
                validate_form_factors(job, form_factor_path(job, directory), system['form_factor_signature'])
            except (OSError, KeyError, ValueError) as error:
                failures.append(f'{system["number"]}: {error}')
        if result.returncode and not failures:
            failures.append(f'The Julia calculation exited with status {result.returncode}.')
    if failures:
        raise ValueError('Job incomplete:\n' + '\n'.join(failures))
    print_stage_timings('Run timings', run_times, perf_counter() - started)
    print(f'\n  Results: {output}', flush=True)


def main():
    """
    Read the single job path supplied to run.py.

    # Arguments:
    - None.
    """
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python run.py jobs/job.yaml')
    try:
        run_job(load_job(sys.argv[1]))
    except (OSError, KeyError, ValueError, yaml.YAMLError) as error:
        raise SystemExit(str(error)) from None
