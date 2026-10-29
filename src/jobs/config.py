"""Read physical calculation parameters from a job YAML."""

import csv
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODES = ('spherical', 'fourier', 'cartesian', 'crystal')
DEFAULTS = {
    'tddft': {'basis': 'def2-TZVPD', 'xc': 'wB97X-D4', 'ring_flatten': False,
              'dft_optimisation': False, 'plot_molecule_3d': False},
    'execution': {'precision': 'float32', 'gpu': False, 'threads': 'auto'},
    'spherical': {'q_max': 20.0, 'n_q': 251, 'l_max': 24, 'n_theta': 0,
                  'n_phi': 0, 'threshold': 1.0e-6, 'save': ['f_s', 'f_sq_lm'],
                  'force_recomputation': False},
    'fourier': {'q_max': [15.0, 15.0, 15.0], 'dq': [0.075, 0.075, 0.075],
                'check_parseval': False},
    'cartesian': {'q_x': [-15.0, 15.0, 101], 'q_y': [-15.0, 15.0, 101],
                  'q_z': [-15.0, 15.0, 101], 'threshold': 1.0e-6, 'save': ['f_s']},
    'crystal': {'order': 'coherent', 'coupling_method': 'ewald', 'coupling_cutoff': 40.0,
                'epsilon_b': 'inf', 'ewald_epsilon_tot': 1.0e-3, 'ewald_eta': 0.0,
                'q_bc': 1.0e-4, 'energy_step': 0.01,
                'band_range_star_stride': 8, 'band_range_q_stride': 4,
                'structure_function': True, 'couplings': True,
                'incoherent_reference': True},
    'verification': {'oscillator_strength': False, 'q_max_fit': 0.05},
}


def check_fields(values, allowed, label):
    """
    Reject unrecognised YAML settings.

    # Arguments:
    - values::dict: The supplied settings.
    - allowed::iterable: The recognised field names.
    - label::str: The settings section.
    """
    if not isinstance(values, dict):
        raise ValueError(f'{label} must be a mapping.')
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f'Unknown {label} setting: {", ".join(sorted(unknown))}.')


def number(value, label, minimum=0, integer=False, strict=False):
    """
    Read a finite physical parameter and check its lower limit.

    # Arguments:
    - value::number: The supplied value.
    - label::str: The parameter name.
    - minimum::number: The lower limit.
    - integer::bool: Whether an integer is required.
    - strict::bool: Whether the lower limit is excluded.

    # Returns:
    - number: The validated value.
    """
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f'{label} must be numeric.') from None
    if isinstance(value, bool) or not math.isfinite(result):
        raise ValueError(f'{label} must be finite and numeric.')
    if result < minimum or (strict and result == minimum):
        raise ValueError(f'{label} must be {">" if strict else ">="} {minimum}.')
    if integer and result != int(result):
        raise ValueError(f'{label} must be an integer.')
    return int(result) if integer else result


def load_job(path):
    """
    Read and validate a job, resolving paths relative to its YAML.

    # Arguments:
    - path::Path: The job YAML.

    # Returns:
    - dict: The resolved calculation settings.
    """
    # Read the job file. Paths inside it are taken relative to the file itself.
    path = Path(path).resolve()
    with path.open() as source:
        job = yaml.safe_load(source)
    check_fields(job, ['mode', 'output', 'input', 'nstates', 'stages', 'transitions', *DEFAULTS], 'job')

    # The method, which decides which settings sections apply further down.
    mode = job.get('mode')
    if mode not in MODES:
        raise ValueError(f'mode must be one of {", ".join(MODES)}.')
    if not isinstance(job.get('output'), str) or not job['output']:
        raise ValueError('Specify the output directory.')
    job['output'] = str((path.parent / job['output']).resolve())

    # Exactly one input, and it has to be the kind the method expects.
    source = job.get('input', {})
    check_fields(source, ['smiles', 'csv', 'cif', 'cif_directory'], 'input')
    if len(source) != 1 or not all(isinstance(v, str) and v for v in source.values()):
        raise ValueError('Specify exactly one input: smiles, csv, cif or cif_directory.')
    key = next(iter(source))
    if (mode == 'crystal') != (key in ('cif', 'cif_directory')):
        raise ValueError('Crystal jobs need CIF input. Molecular jobs need SMILES or CSV input.')
    if key != 'smiles':
        source[key] = str((path.parent / source[key]).resolve())
        if not Path(source[key]).exists():
            raise ValueError(f'Input does not exist: {source[key]}')
    job['nstates'] = number(job.get('nstates'), 'nstates', 1, integer=True)

    # Which of the two stages to run, reuse from a previous run, or leave out.
    stages = job.get('stages', {})
    check_fields(stages, ['tddft', 'form_factors'], 'stages')
    if set(stages) != {'tddft', 'form_factors'} or any(s not in ('run', 'reuse', 'skip') for s in stages.values()):
        raise ValueError('Set both stages.tddft and stages.form_factors to run, reuse or skip.')
    if stages['form_factors'] != 'skip' and stages['tddft'] == 'skip':
        raise ValueError('Form factors require stages.tddft to be run or reuse.')

    # Work out which settings sections this method uses, then fill the gaps with the defaults.
    applicable = {'tddft', 'execution', 'verification'}
    applicable.add('spherical' if mode == 'crystal' else mode)
    if mode == 'crystal':
        applicable.add('crystal')
    for section in DEFAULTS:
        if section in job and section not in applicable:
            raise ValueError(f'{section} settings do not apply to mode {mode}.')
        if section not in applicable:
            continue
        supplied = job.get(section, {})
        check_fields(supplied, DEFAULTS[section], section)
        job[section] = {**DEFAULTS[section], **supplied}
        for name, default in DEFAULTS[section].items():
            if isinstance(default, bool) and not isinstance(job[section][name], bool):
                raise ValueError(f'{section}.{name} must be true or false.')
    # Precision, GPU and thread count.
    execution = job['execution']
    if execution['precision'] not in ('float32', 'float64'):
        raise ValueError('execution.precision must be float32 or float64.')
    if execution['threads'] != 'auto':
        execution['threads'] = number(execution['threads'], 'execution.threads', 1, integer=True)

    # The basis set and functional.
    for name in ('basis', 'xc'):
        if not isinstance(job['tddft'][name], str) or not job['tddft'][name]:
            raise ValueError(f'tddft.{name} must be a nonempty string.')

    # Which transitions to compute form factors for, out of the states TD-DFT saved.
    transitions = job.get('transitions', 'all')
    if transitions != 'all':
        if not isinstance(transitions, list) or not transitions:
            raise ValueError('transitions must be all or a nonempty list of indices.')
        transitions = [number(s, 'transition', 1, integer=True) for s in transitions]
        if len(set(transitions)) != len(transitions) or max(transitions) > job['nstates']:
            raise ValueError('Transition indices must be unique and no greater than nstates.')
    job['transitions'] = transitions
    # The spherical grid, which the crystal path also uses.
    if 'spherical' in job:
        settings = job['spherical']
        for name in ('n_q', 'l_max', 'n_theta', 'n_phi'):
            settings[name] = number(settings[name], 'spherical.' + name,
                                    2 if name == 'n_q' else 0, integer=True)
        for name in ('n_theta', 'n_phi'):
            if settings[name] == 1:
                raise ValueError(f'spherical.{name} must be zero (automatic) or at least two.')
        for name in ('q_max', 'threshold'):
            settings[name] = number(settings[name], 'spherical.' + name, strict=name == 'q_max')
        if not isinstance(settings['save'], list) or not settings['save'] or set(settings['save']) - {'f_s', 'R_tensor', 'f_sq_lm'}:
            raise ValueError('spherical.save selects f_s, R_tensor and/or f_sq_lm.')
    # The Fourier grid, given as a limit and a spacing per axis.
    if mode == 'fourier':
        for name in ('q_max', 'dq'):
            values = job['fourier'][name]
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError(f'fourier.{name} needs three values.')
            job['fourier'][name] = [number(v, 'fourier.' + name, strict=True) for v in values]
        if any(round(2*q/dq) + 1 < 3 for q, dq in zip(job['fourier']['q_max'], job['fourier']['dq'])):
            raise ValueError('The Fourier grid needs at least three points on each axis.')
    # The Cartesian grid, given as a range per axis.
    if mode == 'cartesian':
        for name in ('q_x', 'q_y', 'q_z'):
            values = job['cartesian'][name]
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError(f'cartesian.{name} must be [min, max, count].')
            low, high = float(values[0]), float(values[1])
            if not low < high or not all(math.isfinite(v) for v in (low, high)):
                raise ValueError(f'cartesian.{name} needs finite increasing limits.')
            job['cartesian'][name] = [low, high, number(values[2], name + ' count', 2, integer=True)]
        job['cartesian']['threshold'] = number(job['cartesian']['threshold'], 'cartesian.threshold')
        if not isinstance(job['cartesian']['save'], list) or not job['cartesian']['save'] or set(job['cartesian']['save']) - {'f_s', 'V'}:
            raise ValueError('cartesian.save selects f_s and/or V.')
    # The oscillator strength check needs the unsquared form factor to fit the dipole from.
    verification = job['verification']
    verification['q_max_fit'] = number(verification['q_max_fit'], 'verification.q_max_fit', strict=True)
    if verification['oscillator_strength'] and mode != 'fourier':
        saved = job['spherical' if mode == 'crystal' else mode]['save']
        if 'f_s' not in saved:
            raise ValueError('The oscillator strength check requires f_s in save.')

    # The crystal settings: how to combine the molecules, and how to sum the Coulomb couplings.
    if mode == 'crystal':
        settings = job['crystal']
        if settings['order'] not in ('coherent', 'incoherent') or settings['coupling_method'] not in ('ewald', 'direct'):
            raise ValueError('Choose coherent/incoherent order and ewald/direct couplings.')
        # Inf is the conducting boundary condition, and is carried as a string through the JSON.
        epsilon_b = float(settings['epsilon_b'])
        if not epsilon_b >= 1:
            raise ValueError('crystal.epsilon_b must be at least one.')
        settings['epsilon_b'] = 'inf' if epsilon_b == float('inf') else epsilon_b
        for name in ('coupling_cutoff', 'ewald_eta', 'ewald_epsilon_tot', 'q_bc', 'energy_step'):
            settings[name] = number(settings[name], 'crystal.' + name,
                                    strict=name not in ('coupling_cutoff', 'ewald_eta'))
        if settings['ewald_epsilon_tot'] >= 1:
            raise ValueError('crystal.ewald_epsilon_tot must be less than one.')
        for name in ('band_range_star_stride', 'band_range_q_stride'):
            settings[name] = number(settings[name], 'crystal.' + name, 1, integer=True)
    return job


def read_systems(job):
    """
    Number molecules or crystals in a deterministic input order.

    # Arguments:
    - job::dict: The resolved job settings.

    # Returns:
    - list[dict]: The numbered systems and their inputs.
    """
    # Gather the inputs, whichever of the four kinds was given.
    source = job['input']
    if 'smiles' in source:
        values = [source['smiles']]
    elif 'csv' in source:
        with open(source['csv'], newline='') as handle:
            reader = csv.DictReader(handle)
            if 'smiles' not in (reader.fieldnames or []):
                raise ValueError('The molecular CSV needs a smiles column.')
            values = [row['smiles'].strip() for row in reader]
        if not all(values):
            raise ValueError('The CSV contains an empty SMILES value.')
    elif 'cif' in source:
        values = [source['cif']]
    else:
        values = [str(path) for path in sorted(Path(source['cif_directory']).glob('*'))
                  if path.is_file() and path.suffix.lower() == '.cif']
    if not values:
        raise ValueError('The input contains no molecules or crystals.')

    # Number them from one, which is what the output directories are named after.
    return [{'number': i, 'input': value,
             'label': Path(value).stem if job['mode'] == 'crystal' else value}
            for i, value in enumerate(values, 1)]
