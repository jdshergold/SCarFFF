"""Draw the individual molecular and structure function plots."""

import numpy as np
from scipy.integrate import trapezoid
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.ticker import MaxNLocator

from . import style


def density_half_width(horizontal, vertical, values, fraction=0.01):
    """
    Find the half width of a plane that holds all of the transition density worth seeing.

    # Arguments:
    - horizontal::np.ndarray: The horizontal coordinates, in Angstroms.
    - vertical::np.ndarray: The vertical coordinates, in Angstroms.
    - values::np.ndarray: The plane values, in Angstrom^-3.
    - fraction::float: The share of the peak below which the density is treated as empty.

    # Returns:
    - float: The half width to draw out to, in Angstroms.
    """
    # The FFT box is set by the momentum spacing, so it is usually far larger than the molecule and
    # a plane drawn over all of it is mostly blank. Keep whatever rises above a small share of the
    # peak, and draw a square window around it so the molecule stays centred.
    limit = np.max(np.abs(values))
    rows, columns = np.nonzero(np.abs(values) > fraction * limit)
    if limit <= 0 or rows.size == 0:
        return max(abs(horizontal[0]), abs(horizontal[-1]), abs(vertical[0]), abs(vertical[-1]))
    half_width = max(np.max(np.abs(horizontal[columns])), np.max(np.abs(vertical[rows])))

    # A margin of one grid step, so the outermost points are not clipped by the frame edge.
    step = max(horizontal[1] - horizontal[0], vertical[1] - vertical[0])
    return float(half_width + step)


def state_subscript(states):
    """
    Format an individual state or an intensity group.

    # Arguments:
    - states::tuple: The original transition indices.

    # Returns:
    - str: The LaTeX subscript.
    """
    members = ','.join(str(s) for s in states)
    return r'\{' + members + r'\}' if len(states) > 1 else members


def line_style(index):
    """
    Distinguish each curve by colour and line style.

    # Arguments:
    - index::int: The zero-based curve index.

    # Returns:
    - dict: The line colour, pattern and width.
    """
    # Cycle the colours, and step the line pattern each time the colours come back round, so curves
    # stay distinguishable past the end of the palette and in black and white.
    return dict(color=style.PALETTE[index % len(style.PALETTE)],
                ls=style.LINESTYLES[(index + index // len(style.PALETTE)) % len(style.LINESTYLES)], lw=1.1)


def mode_key(mode, n_modes):
    """
    Find a saved spherical harmonic's zero-based key.

    # Arguments:
    - mode::tuple: The indices (l, m).
    - n_modes::int: The number of available spherical harmonics.

    # Returns:
    - int: The spherical harmonic key.
    """
    # The spherical harmonics are stored using the key l^2 + l + m.
    l, m = mode
    key = l*l + l + m
    if l < 0 or abs(m) > l or key >= n_modes:
        raise ValueError(f'The spherical harmonic ({l},{m}) was not saved.')
    return key


def molecular_plot(data, request, settings):
    """
    Draw a molecular slice, coefficient plot or angular moment comparison.

    # Arguments:
    - data::MolecularData: The saved molecular quantities.
    - request::dict: The figure selection.
    - settings::PrintSettings: The final printed size and output directory.
    """
    kind = request['type']
    if kind == 'f_sq_lm' and isinstance(request['states'], str):
        data.require(data.energies, 'f_sq_lm')
    groups = data.select_groups(request['states'], request['count'])

    # The real and imaginary parts cannot be added across a degenerate group, since the phases are
    # arbitrary, so those plots take one transition at a time. Nor can the transition density, which
    # is signed for the same reason.
    if kind in ('real', 'imag', 'transition_density'):
        states = request['states']
        if isinstance(states, list):
            groups = [(s,) for s in states]
        else:
            groups = [(s,) for group in groups for s in group]

    # A plane through the form factor, either its intensity or one signed part of it.
    if kind in ('intensity', 'real', 'imag'):
        for group in groups:
            for plane in request['planes']:
                horizontal, vertical, values = data.plane(group, plane, kind)
                fig, ax = style.make_frame(settings, square=True)
                # A signed quantity gets a diverging colour map centred on zero, an intensity does not.
                signed = kind != 'intensity'
                limit = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
                norm = Normalize(-limit if signed else 0, limit)
                cmap = 'RdBu_r' if signed else 'magma'
                ax.imshow(values, origin='lower', cmap=cmap, norm=norm, interpolation='nearest',
                          extent=(horizontal[0], horizontal[-1], vertical[0], vertical[-1]), rasterized=True)
                # The same range on both axes, so the plane is not distorted.
                low = min(horizontal[0], vertical[0])
                high = max(horizontal[-1], vertical[-1])
                ax.set_xlim(low, high)
                ax.set_ylim(low, high)
                ax.set_aspect('equal')
                ax.grid(False, which='both')
                ax.set_xlabel(rf'$q_{plane[0]}\ [\mathrm{{keV}}]$')
                ax.set_ylabel(rf'$q_{plane[1]}\ [\mathrm{{keV}}]$')
                subscript = state_subscript(group)
                if signed:
                    symbol = r'\operatorname{Re}' if kind == 'real' else r'\operatorname{Im}'
                    label = rf'${symbol}[f_{{{subscript}}}(\vec q)]$'
                else:
                    label = rf'$|f_{{{subscript}}}(\vec q)|^2$'
                style.add_colourbar(fig, ax, norm, cmap, label, settings, signed=signed)
                fixed = {'xy': 'z', 'xz': 'y', 'yz': 'x'}[plane]
                style.add_boxes(ax, [rf'$q_{fixed}=0$'])
                name = f'{kind}_s' + '-'.join(map(str, group)) + '_' + plane
                style.save_plot(fig, settings, name, square=True)
        return
    # A plane through the transition density, drawn like the form factor planes but in real space.
    if kind == 'transition_density':
        for group in groups:
            for plane in request['planes']:
                horizontal, vertical, values = data.density_plane(group[0], plane)
                fig, ax = style.make_frame(settings, square=True)
                # The density is signed, since the excitation moves charge from one place to another.
                limit = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
                norm = Normalize(-limit, limit)
                ax.imshow(values, origin='lower', cmap='RdBu_r', norm=norm, interpolation='nearest',
                          extent=(horizontal[0], horizontal[-1], vertical[0], vertical[-1]), rasterized=True)
                # The same range on both axes, so the molecule is not distorted, and only as far out
                # as the density reaches rather than the whole FFT box.
                half_width = density_half_width(horizontal, vertical, values)
                ax.set_xlim(-half_width, half_width)
                ax.set_ylim(-half_width, half_width)
                ax.set_aspect('equal')
                ax.grid(False, which='both')
                ax.set_xlabel(rf'${plane[0]}\ [\mathrm{{\AA}}]$')
                ax.set_ylabel(rf'${plane[1]}\ [\mathrm{{\AA}}]$')
                style.add_colourbar(fig, ax, norm, 'RdBu_r',
                    rf'$\Phi^{{g\to {group[0]}}}(\vec r)\ [\mathrm{{\AA}}^{{-3}}]$', settings, signed=True)
                fixed = {'xy': 'z', 'xz': 'y', 'yz': 'x'}[plane]
                style.add_boxes(ax, [rf'${fixed}=0$'])
                name = f'{kind}_s{group[0]}_{plane}'
                style.save_plot(fig, settings, name, square=True)
        return
    # The intensity coefficients against q, one line per spherical harmonic.
    if kind == 'f_sq_lm':
        for group in groups:
            coefficients = data.coefficients(group)
            # Rank the spherical harmonics by the area under |f^2_lm(q)|, and take the strongest.
            if request['modes'] == 'dominant':
                areas = trapezoid(np.abs(coefficients), data.grid['q'], axis=0)
                keys = np.argsort(areas)[::-1][:request['count']]
                modes = []
                for key in keys:
                    # Invert key = l^2 + l + m to get back (l, m).
                    l = int(np.sqrt(key))
                    modes.append((l, int(key-l*l-l)))
            else:
                modes = request['modes']
            fig, ax = style.make_frame(settings)
            for i, mode in enumerate(modes):
                key = mode_key(mode, coefficients.shape[1])
                ax.plot(data.grid['q'], coefficients[:, key], label=rf'$({mode[0]},{mode[1]})$', **line_style(i))
            ax.axhline(0, color='.45', lw=.5)
            ax.set_xlim(data.grid['q'][0], data.grid['q'][-1])
            ax.set_xlabel(r'$q\ [\mathrm{keV}]$')
            ax.set_ylabel(rf'$f^2_{{{state_subscript(group)},\ell m}}(q)$')
            style.add_line_legend(ax, settings, 'Dominant Modes' if request['modes'] == 'dominant' else 'Angular Modes')
            style.save_plot(fig, settings, 'f_sq_lm_s' + '-'.join(map(str, group)))
        return
    # Otherwise the angular mean or spread of the intensity, one line per transition group.
    fig, ax = style.make_frame(settings)
    for i, group in enumerate(groups):
        mean, deviation = data.moments(group)
        values = mean if kind == 'angular_mean' else deviation
        ax.plot(data.angular_grid[0], values, label=rf'$s={state_subscript(group)}$', **line_style(i))
    ax.set_xlim(0, data.angular_grid[0][-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$q\ [\mathrm{keV}]$')
    ax.set_ylabel(r'$\langle |f_s(q)|^2\rangle_\Omega$' if kind == 'angular_mean' else r'$\sigma_\Omega(|f_s|^2)$')
    style.add_line_legend(ax, settings, 'Dominant Transitions' if request['states'] == 'dominant' else 'Transitions')
    style.save_plot(fig, settings, kind)


def structure_plot(data, request, settings):
    """
    Draw structure function maps or energy-integrated coefficients.

    # Arguments:
    - data::CrystalData: The normalised crystal output.
    - request::dict: The figure selection.
    - settings::PrintSettings: The final printed size and output directory.
    """
    kind = request['type']

    # The angular average, integrated over energy, as a single curve against q.
    if kind == 'structure_integral':
        fig, ax = style.make_frame(settings)
        ax.plot(data.q, data.integrated_mean, **line_style(0))
        ax.set_xlim(data.q[0], data.q[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$q\ [\mathrm{keV}]$')
        ax.set_ylabel(r'$\int dE\,\langle S(\vec q,E)\rangle_\Omega\ [\mathrm{\AA}^{-3}]$')
        style.save_plot(fig, settings, kind)
        return
    modes = data.ranked_modes(request['count']) if request['modes'] == 'dominant' else request['modes']

    # One curve per spherical harmonic. The sum rule plot is a check rather than a result: summing
    # over the crystal states must give back the incoherent sum of the same molecules, because the
    # mixing coefficients are unitary, so these curves should sit at zero to machine precision.
    if kind in ('structure_mode_integrals', 'structure_sum_rule'):
        if kind == 'structure_sum_rule':
            coefficients = data.integrals - data.incoherent_sums
            label = r'$\int dE\,(S_{\ell m}-S_{\ell m}^{(0)})\ [\mathrm{\AA}^{-3}]$'
        else:
            coefficients = data.integrals
            label = r'$\int dE\,S_{\ell m}(q,E)\ [\mathrm{\AA}^{-3}]$'
        fig, ax = style.make_frame(settings)
        for i, mode in enumerate(modes):
            key = mode_key(mode, data.n_modes)
            ax.plot(data.q, coefficients[:, key], label=rf'$({mode[0]},{mode[1]})$', **line_style(i))
        ax.axhline(0, color='.45', lw=.5)
        ax.set_xlim(data.q[0], data.q[-1])
        ax.set_xlabel(r'$q\ [\mathrm{keV}]$')
        ax.set_ylabel(label)
        style.add_line_legend(ax, settings, 'Dominant Modes' if request['modes'] == 'dominant' else 'Angular Modes')
        style.save_plot(fig, settings, kind)
        return
    # The (q, E) maps. Several harmonics share one figure, so how many fit depends on the width.
    if kind == 'structure_modes':
        count = {'full': 6, 'half': 2, 'third': 1}[settings.width]
        if request['modes'] == 'dominant':
            modes = data.ranked_modes(count)
        if len(modes) != count:
            raise ValueError(f'The {settings.width}-width mode layout needs {count} modes.')
        selections = [modes]
    else:
        selections = [[mode] for mode in modes]
    for modes in selections:
        values = [data.mode(mode) for mode in modes]
        limit = max(max(float(np.max(np.abs(value))) for value in values), np.finfo(float).tiny)

        # Only the monopole is positive everywhere, so it gets a plain log scale. The rest change
        # sign, so they need a symmetric log scale with a linear region around zero.
        monopole = len(modes) == 1 and tuple(modes[0]) == (0, 0)
        norm = LogNorm(limit*1e-4, limit, clip=True) if monopole else SymLogNorm(
            linthresh=limit*1e-4, linscale=.6, vmin=-limit, vmax=limit, base=10)
        cmap = 'magma' if monopole else 'RdBu_r'

        # Share one colour scale across the panels, so they can be compared by eye.
        fig, ax = style.make_frame(settings)
        position = ax.get_position()
        axes = [ax] + [fig.add_axes(position.bounds) for i in range(1, len(modes))]
        style.place_frames(fig, (position.x0*fig.get_figwidth(), position.y0*fig.get_figheight(),
                                 position.width*fig.get_figwidth(), position.height*fig.get_figheight()))
        rows = 3 if len(modes) == 6 else len(modes)
        for i, (ax, mode, value) in enumerate(zip(axes, modes, values)):
            ax.pcolormesh(data.q, data.energy, value, cmap=cmap, norm=norm, shading='nearest', rasterized=True)
            ax.set_xlim(data.q[0], data.q[-1])
            ax.set_ylim(data.energy[0], data.energy[-1])
            style.style_axes(ax, heatmap=True, tick_size=settings.caption_pt)
            # Label only the outer edges, so the inner panels are not cluttered.
            row, column = i % rows, i // rows
            ax.tick_params(labelbottom=row == rows-1, labelleft=column == 0)
            if row == rows-1:
                ax.set_xlabel(r'$q\ [\mathrm{keV}]$')
            if i == 0:
                ax.set_ylabel(r'$E\ [\mathrm{eV}]$')
            x_bins = 4 if settings.width == 'third' else 6 if settings.width == 'half' or len(modes) > 1 else 10
            y_bins = 4 if settings.width == 'third' else 5 if len(modes) > 1 else 6
            ax.xaxis.set_major_locator(MaxNLocator(nbins=x_bins, steps=[1, 2, 2.5, 5, 10]))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=y_bins, steps=[1, 2, 2.5, 5, 10], prune='both' if len(modes) > 1 else None))
            if len(modes) > 1:
                style.add_boxes(ax, [rf'$({mode[0]},{mode[1]})$'])
        # A negative m uses a comma, since S_{2-1} reads as a subtraction.
        l, m = modes[0]
        symbol = r'S_{\ell m}' if len(modes) > 1 else rf'S_{{{l},{m}}}' if m < 0 else rf'S_{{{l}{m}}}'
        label = '$' + symbol + r'(q,E)\ [\mathrm{\AA}^{-3}\,\mathrm{eV}^{-1}]$'
        style.place_frames(fig, (position.x0*fig.get_figwidth(), position.y0*fig.get_figheight(),
                                 position.width*fig.get_figwidth(), position.height*fig.get_figheight()))
        style.add_colourbar(fig, axes[0], norm, cmap, label, settings, signed=not monopole)
        name = 'structure_modes' if len(modes) > 1 else f'S_l{modes[0][0]}_m{modes[0][1]}'
        style.save_plot(fig, settings, name)
