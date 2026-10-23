"""Select saved systems and draw only the figures requested in the plotting YAML."""

from pathlib import Path
import sys

import yaml

from .config import DEFAULT_CONFIG, figures_for_data, figures_for_mode, load_plot_config
from .data import CrystalData, MolecularData


def result_directories(path, selection):
    """
    Select numbered molecular or crystal directories from a run.

    # Arguments:
    - path::Path: A run or numbered result directory.
    - selection::str or list: all or a list of original system numbers.

    # Returns:
    - list[Path]: The selected result directories.
    """
    # A single numbered directory can be given directly, rather than the whole run.
    if (path / 'form_factors.h5').exists() or (path / 'crystal').is_dir():
        return [path]
    if selection == 'all':
        directories = sorted([child for child in path.iterdir() if child.is_dir() and child.name.isdigit()],
                             key=lambda child: int(child.name))
    else:
        directories = [path / str(number) for number in selection]
    return directories


def plot_run(path, config):
    """
    Draw the selected figures for each saved molecule or crystal.

    # Arguments:
    - path::Path: The run directory.
    - config::dict: The resolved plotting settings.
    """
    from . import style
    directories = result_directories(Path(path).resolve(), config['molecules'])
    if not directories:
        raise ValueError('No numbered molecular or crystal result directories were found.')
    # Carry on to the next system after a failure, so one bad result does not stop the rest.
    failures = []
    for directory in directories:
        try:
            plot_system(directory, config)
        except (OSError, KeyError, ValueError) as error:
            style.plt.close('all')
            failures.append(f'{directory.name}: {error}')
    if failures:
        raise ValueError('Plotting incomplete:\n' + '\n'.join(failures))


def plot_system(directory, config):
    """
    Draw the requested figures for one numbered molecule or crystal.

    # Arguments:
    - directory::Path: The saved system directory.
    - config::dict: The resolved plotting settings.
    """
    from . import style
    from .figures import molecular_plot, structure_plot
    from .overviews import molecular_overview, structure_overview
    # The saved job says which method produced this, which decides what can be drawn.
    molecule = directory / 'form_factors.h5'
    job_path = directory.parent / 'job.yaml'
    job = yaml.safe_load(job_path.read_text()) if job_path.exists() else {}
    if molecule.exists() and job.get('mode') != 'crystal':
        data = MolecularData(molecule, config['grouping']['energy_tolerance_eV'], config['angular_sampling'])
        requests = figures_for_mode(config['figures'], data.mode)
    else:
        # A crystal directory can hold both precisions, so the job picks the one that belongs.
        candidates = list((directory / 'crystal').glob('crystal_form_factors_*.h5'))
        if len(candidates) > 1 and job:
            precision = job['execution']['precision']
            suffix = 'f32' if precision == 'float32' else 'f64'
            candidates = [candidate for candidate in candidates if candidate.name.endswith(f'_{suffix}.h5')]
        if len(candidates) != 1:
            raise ValueError(f'{directory}: expected one current crystal file or form_factors.h5.')
        requests = figures_for_mode(config['figures'], 'crystal')
        if not requests:
            return
        data = CrystalData(candidates[0])
        requests = figures_for_data(requests, data.saved)
    if not requests:
        return

    # Draw each requested figure, grouping them by page width so each width shares its settings.
    output = directory / 'plots'
    output.mkdir(exist_ok=True)
    style.use_style(config['style']['caption_pt'])
    widths = {}
    for request in requests:
        # The overview is one standalone figure, so it is saved on its own rather than batched.
        if request['type'] == 'overview':
            fig = molecular_overview(data, request, config) if isinstance(data, MolecularData) else structure_overview(data, config)
            name = 'molecular_overview' if isinstance(data, MolecularData) else 'structure_function_overview'
            for extension in config['file_type']:
                fig.savefig(output / f'{name}.{extension}', dpi=220)
            style.plt.close(fig)
            continue
        width = request['width']
        if width not in widths:
            destination = output / width
            destination.mkdir(exist_ok=True)
            widths[width] = style.PrintSettings(width=width, output_dir=destination,
                                               file_type=tuple(config['file_type']), **config['style'])
        settings = widths[width]
        settings.shape = request['shape']
        if isinstance(data, MolecularData):
            molecular_plot(data, request, settings)
        else:
            structure_plot(data, request, settings)
    for settings in widths.values():
        style.save_figures(settings)
    print(f'Plots saved to {output}')


def main():
    """
    Read the result path and optional plotting YAML supplied to plot.py.

    # Arguments:
    - None.
    """
    if len(sys.argv) not in (2, 3):
        raise SystemExit('Usage: python plot.py runs/job [config/plotting.yaml]')
    try:
        config = load_plot_config(sys.argv[2] if len(sys.argv) == 3 else DEFAULT_CONFIG)
        plot_run(Path(sys.argv[1]), config)
    except (OSError, KeyError, ValueError, yaml.YAMLError) as error:
        raise SystemExit(str(error)) from None
