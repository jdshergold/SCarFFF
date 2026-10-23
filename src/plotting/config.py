"""Read the persistent plotting configuration."""

from pathlib import Path

import yaml

from ..jobs.config import MODES, check_fields, number

MOLECULAR_MODES = {'spherical', 'cartesian', 'fourier'}
FIGURE_MODES = {
    'overview': set(MODES),
    'intensity': MOLECULAR_MODES,
    'real': MOLECULAR_MODES,
    'imag': MOLECULAR_MODES,
    'f_sq_lm': {'spherical'},
    'transition_density': {'fourier'},
    'angular_mean': MOLECULAR_MODES,
    'angular_std': MOLECULAR_MODES,
    'structure_mode': {'crystal'},
    'structure_modes': {'crystal'},
    'structure_integral': {'crystal'},
    'structure_mode_integrals': {'crystal'},
    'structure_sum_rule': {'crystal'},
}
# What a figure needs in the file on top of the right calculation mode. The sum rule weighs the
# mixed crystal states against the same molecules unmixed, so it needs both of those saved.
FIGURE_DATASETS = {'structure_sum_rule': ('S_lm', 'f_sq_lm_incoherent')}
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'config/plotting.yaml'


def figures_for_mode(figures, mode):
    """
    Select figure requests that apply to the calculation method.

    # Arguments:
    - figures::list: The validated figure requests.
    - mode::str: The calculation method.

    # Returns:
    - list: The applicable requests, in configuration order.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown calculation mode: {mode}.')
    return [request for request in figures if mode in FIGURE_MODES[request['type']]]


def figures_for_data(figures, saved):
    """
    Drop figure requests whose quantities the run was not asked to save.

    # Arguments:
    - figures::list: The applicable figure requests.
    - saved::set: The names of the saved datasets.

    # Returns:
    - list: The requests that can be drawn, in configuration order.
    """
    # One request expands into a figure per width and shape, so collect the missing quantities by
    # figure type and say something once rather than once per layout.
    kept = []
    skipped = {}
    for request in figures:
        missing = [name for name in FIGURE_DATASETS.get(request['type'], ()) if name not in saved]
        if missing:
            skipped[request['type']] = missing
        else:
            kept.append(request)
    for kind, missing in skipped.items():
        print(f'Skipping {kind}: {", ".join(missing)} was not saved.')
    return kept


def plot_choices(value, allowed, name):
    """
    Read one plotting option or a list of distinct options.

    # Arguments:
    - value::str or list: The requested options.
    - allowed::tuple: The recognised options.
    - name::str: The configuration field.

    # Returns:
    - list: The requested options, in configuration order.
    """
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not values or any(option not in allowed for option in values):
        raise ValueError(f'{name} must select {", ".join(allowed)}, individually or as a list.')
    if len(set(values)) != len(values):
        raise ValueError(f'List each {name} option only once.')
    return values


def load_plot_config(path=DEFAULT_CONFIG):
    """
    Validate the print style and the explicit figure requests.

    # Arguments:
    - path::Path: The plotting YAML.

    # Returns:
    - dict: The resolved plotting settings.
    """
    # Read the file, then fill in the defaults section by section.
    with open(path) as source:
        config = yaml.safe_load(source)
    check_fields(config, ['file_type', 'width', 'shape', 'style', 'grouping', 'angular_sampling', 'molecules', 'figures'], 'plotting')

    # The image format, and the page widths and shapes a figure may be drawn at.
    config['file_type'] = plot_choices(config.get('file_type', ['png', 'pdf']), ('png', 'pdf'), 'file_type')
    config['width'] = plot_choices(config.get('width', ['full', 'half', 'third']),
                                    ('full', 'half', 'third'), 'width')
    config['shape'] = plot_choices(config.get('shape', ['square', 'rectangular']),
                                    ('square', 'rectangular'), 'shape')
    # Set font sizes.
    style = config.get('style', {})
    check_fields(style, ['caption_pt', 'text_width_pt', 'gap_pt'], 'style')
    config['style'] = {'caption_pt': 9, 'text_width_pt': 510, 'gap_pt': 18, **style}
    for name in config['style']:
        config['style'][name] = number(config['style'][name], 'style.' + name, strict=name != 'gap_pt')
    if config['style']['text_width_pt'] <= 2 * config['style']['gap_pt']:
        raise ValueError('The text width must exceed the two interfigure gaps.')
    # How close in energy two transitions must be to be drawn as one degenerate group.
    grouping = config.get('grouping', {})
    check_fields(grouping, ['energy_tolerance_eV'], 'grouping')
    config['grouping'] = {'energy_tolerance_eV': number(grouping.get('energy_tolerance_eV', .01), 'energy tolerance')}

    # The grid the coefficients are resampled onto when a figure needs the angles back.
    sampling = config.get('angular_sampling', {})
    check_fields(sampling, ['n_q', 'n_theta', 'n_phi'], 'angular_sampling')
    config['angular_sampling'] = {'n_q': 251, 'n_theta': 97, 'n_phi': 97, **sampling}
    for name in config['angular_sampling']:
        config['angular_sampling'][name] = number(config['angular_sampling'][name], name, 3, integer=True)
    # Which of the saved systems to draw.
    config.setdefault('molecules', 'all')
    if config['molecules'] != 'all':
        if not isinstance(config['molecules'], list) or not config['molecules']:
            raise ValueError('molecules must be all or a nonempty list of molecule numbers.')
        config['molecules'] = [number(n, 'molecule number', 1, integer=True) for n in config['molecules']]
    # Then the figures themselves. One request can expand into several, one per width and shape.
    figures = config.get('figures')
    if not isinstance(figures, list) or not figures:
        raise ValueError('Specify a nonempty figures list.')
    resolved = []
    for supplied in figures:
        check_fields(supplied, ['type', 'width', 'shape', 'states', 'modes', 'count', 'planes'], 'figure')
        kind = supplied.get('type')
        if kind not in FIGURE_MODES:
            raise ValueError(f'Unknown figure type: {kind}.')
        widths = plot_choices(supplied.get('width', 'full' if kind == 'overview' else config['width']),
                              ('full', 'half', 'third'), 'width')
        shapes = plot_choices(supplied.get('shape', config['shape']), ('square', 'rectangular'), 'shape')
        # The overview only exists full width, and the form factor slice plots are always square.
        layouts = []
        if kind == 'overview':
            if 'full' in widths:
                layouts.append(('full', 'rectangular'))
        else:
            for width in widths:
                square_frame = width == 'third' or kind in ('intensity', 'real', 'imag', 'transition_density')
                selected_shapes = ['square'] if square_frame and 'shape' not in supplied else shapes
                for shape in selected_shapes:
                    if square_frame and shape != 'square':
                        continue
                    layouts.append((width, shape))
        if not layouts:
            raise ValueError(f'The requested widths and shapes have no supported layout for {kind}.')
        # Which transitions to draw, either all of them, the strongest, or named indices.
        count = number(supplied['count'], 'figure count', 1, integer=True) if 'count' in supplied else None
        states = supplied.get('states', 'all' if kind in ('real', 'imag') else 'dominant')
        valid_states = states in ('all', 'dominant') if isinstance(states, str) else isinstance(states, list) and bool(states)
        if not valid_states:
            raise ValueError('states must be all, dominant or a nonempty list of original indices.')
        if isinstance(states, list):
            states = [number(s, 'state', 1, integer=True) for s in states]
        # Which (l, m) spherical harmonics to draw, either the strongest or named pairs.
        modes = supplied.get('modes', 'dominant')
        if modes != 'dominant':
            if not isinstance(modes, list) or not modes:
                raise ValueError('modes must be dominant or a nonempty list of [l,m] pairs.')
            for pair in modes:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError('The [l,m] pair must be specified for each spherical harmonic mode.')
                l = number(pair[0], 'l', 0, integer=True)
                m = pair[1]
                if isinstance(m, bool) or not isinstance(m, int) or abs(m) > l:
                    raise ValueError('A spherical harmonic needs integer indices with |m| <= l.')
        # Which Cartesian planes to slice through.
        planes = supplied.get('planes', ['xy'] if kind == 'overview' else ['yz'])
        if not isinstance(planes, list) or not planes or any(plane not in ('xy', 'xz', 'yz') for plane in planes):
            raise ValueError('planes must be a nonempty list drawn from xy, xz and yz.')

        # One resolved request per layout, with the defaults filled in.
        for width, shape in layouts:
            default_count = 8 if width == 'full' else 4
            resolved.append({**supplied, 'width': width, 'shape': shape,
                             'count': count if count is not None else default_count,
                             'states': states, 'modes': modes, 'planes': planes})
    config['figures'] = resolved
    if sum(request['type'] == 'overview' for request in resolved) > 1:
        raise ValueError('Select one overview per plotting configuration.')
    return config
