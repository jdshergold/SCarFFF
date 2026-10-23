"""Draw the four-panel molecular and crystal overviews."""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import numpy as np

from . import style
from .figures import line_style, state_subscript


def overview_colourbar(fig, ax, norm, cmap, label):
    """
    Put a local colourbar immediately above one overview panel.

    # Arguments:
    - fig::matplotlib.figure.Figure: The overview.
    - ax::matplotlib.axes.Axes: The panel axes.
    - norm::matplotlib.colors.Normalize: The absolute colour scale.
    - cmap::str: The colour map.
    - label::str: The quantity and its units.
    """
    # Sit the bar just above the panel, matching its width.
    position = ax.get_position()
    height = fig.get_figheight()
    cax = fig.add_axes((position.x0, position.y1 + .055/height, position.width, .055/height))

    # A symmetric log scale needs its ticks placing by hand, one decade either side of zero.
    ticks = None
    if isinstance(norm, SymLogNorm):
        power = np.floor(np.log10(norm.vmax))
        positive = 10.0 ** np.array([power-2, power])
        ticks = np.concatenate([-positive[::-1], [0], positive])
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation='horizontal', ticks=ticks)
    if ticks is not None:
        bar.formatter = style.FuncFormatter(lambda value, position: '$0$' if value == 0 else
            rf'${"-" if value < 0 else ""}10^{{{int(round(np.log10(abs(value))))}}}$')
        bar.update_ticks()
    bar.ax.xaxis.set_ticks_position('top')
    bar.ax.xaxis.set_label_position('top')
    bar.ax.tick_params(length=2.5, pad=1.5)
    bar.outline.set_linewidth(.6)
    bar.set_label(label, labelpad=5)


def molecular_overview(data, request, config):
    """
    Draw two molecular slices, angular means and absolute angular standard deviations.

    # Arguments:
    - data::MolecularData: The saved molecule.
    - request::dict: The transition and plane selection.
    - config::dict: The plotting settings.

    # Returns:
    - matplotlib.figure.Figure: The overview.
    """
    groups = data.select_groups(request['states'], 3)
    if not groups:
        raise ValueError('The molecular overview needs at least one transition group.')
    moments = [data.moments(group) for group in groups]

    # A square page-width figure, with the four panels placed by hand so they line up.
    width = config['style']['text_width_pt'] / 72.27
    font = config['style']['caption_pt']
    fig = plt.figure(figsize=(width, width))
    fig.text(.5, .978, 'Molecular Form Factor Overview', ha='center', va='top', fontsize=font*1.65)
    rects = [(.135, .535, .335, .335), (.615, .535, .335, .335),
             (.135, .075, .335, .335), (.615, .075, .335, .335)]
    axes = [fig.add_axes(rect) for rect in rects]
    for ax in axes:
        style.style_axes(ax, tick_size=font)

    # The top two panels are intensity planes, each with its own colour scale.
    for i in range(2):
        group = groups[min(i, len(groups)-1)]
        plane = request['planes'][i % len(request['planes'])]
        horizontal, vertical, values = data.plane(group, plane)
        limit = max(float(values.max()), np.finfo(float).tiny)
        norm = Normalize(0, limit)
        ax = axes[i]
        ax.imshow(values, origin='lower', cmap='magma', norm=norm, interpolation='nearest',
                  extent=(horizontal[0], horizontal[-1], vertical[0], vertical[-1]), rasterized=True)
        low = min(horizontal[0], vertical[0])
        high = max(horizontal[-1], vertical[-1])
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False, which='both')
        ax.set_xlabel(rf'$q_{plane[0]}\ [\mathrm{{keV}}]$')
        ax.set_ylabel(rf'$q_{plane[1]}\ [\mathrm{{keV}}]$')
        label = rf'$|f_{{{state_subscript(group)}}}(\vec q)|^2'
        if len(group) > 1:
            label += '=' + '+'.join(rf'|f_{{{s}}}|^2' for s in group)
        overview_colourbar(fig, ax, norm, 'magma', label + '$')
        fixed = {'xy': 'z', 'xz': 'y', 'yz': 'x'}[plane]
        style.add_boxes(ax, [rf'$q_{fixed}=0$'])
    # The bottom two are the angular mean and spread, one line per transition group.
    for i, (group, (mean, deviation)) in enumerate(zip(groups, moments)):
        label = rf'$s={state_subscript(group)}$'
        axes[2].plot(data.angular_grid[0], mean, label=label, **line_style(i))
        axes[3].plot(data.angular_grid[0], deviation, label=label, **line_style(i))
    for ax in axes[2:]:
        ax.set_xlim(0, data.angular_grid[0][-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$q\ [\mathrm{keV}]$')
        ax.legend(loc='best', borderpad=.35)
    axes[2].set_ylabel(r'$\langle |f_s(q)|^2\rangle_\Omega=(1/4\pi)\int d\Omega\,|f_s(\vec q)|^2$')
    axes[3].set_ylabel(r'$\sigma_\Omega(|f_s|^2)$')
    return fig


def structure_overview(data, config):
    """
    Draw the angular mean, its energy integral and three dominant non-monopole modes.

    # Arguments:
    - data::CrystalData: The saved normalised structure function.
    - config::dict: The plotting settings.

    # Returns:
    - matplotlib.figure.Figure: The overview.
    """
    modes = data.ranked_modes(3)
    if len(modes) < 3:
        raise ValueError('The structure function overview needs three non-monopole spherical harmonics.')
    maps = [data.mode(mode) for mode in modes]
    mean = data.angular_mean
    integrated_mean = data.integrated_mean

    # The angular mean is positive, so a plain log scale. The other harmonics change sign, so they
    # share a symmetric log scale instead.
    mean_limit = max(float(mean.max()), np.finfo(float).tiny)
    mean_norm = LogNorm(mean_limit*1e-4, mean_limit, clip=True)
    limit = max(max(float(np.max(np.abs(value))) for value in maps), np.finfo(float).tiny)
    mode_norm = SymLogNorm(linthresh=limit*1e-4, linscale=.6, vmin=-limit, vmax=limit, base=10)

    # Four panels on a page-width figure, placed by hand.
    width = config['style']['text_width_pt'] / 72.27
    font = config['style']['caption_pt']
    fig = plt.figure(figsize=(width, width*8.1/10.4))
    fig.text(.5, .965, 'Structure Function Overview', ha='center', va='top', fontsize=font*1.65)
    rects = [(.105, .515, .365, .285), (.585, .515, .365, .285),
             (.105, .12, .365, .285), (.585, .12, .365, .285)]
    ax_mean = fig.add_axes(rects[0])
    ax_total = fig.add_axes(rects[2])
    ax_integrals = fig.add_axes(rects[3])

    # Top left: the angular mean against q and E.
    ax_mean.pcolormesh(data.q, data.energy, mean, norm=mean_norm,
                       cmap='magma', shading='nearest', rasterized=True)
    ax_mean.set_xlim(data.q[0], data.q[-1])
    ax_mean.set_ylim(data.energy[0], data.energy[-1])
    ax_mean.set_xlabel(r'$q\ [\mathrm{keV}]$')
    ax_mean.set_ylabel(r'$E\ [\mathrm{eV}]$')
    style.style_axes(ax_mean, heatmap=True, tick_size=font)
    overview_colourbar(fig, ax_mean, mean_norm, 'magma',
        r'$\langle S(\vec q,E)\rangle_\Omega=S_{00}(q,E)/\sqrt{4\pi}$' + '\n' +
        r'$[\mathrm{\AA}^{-3}\,\mathrm{eV}^{-1}]$')
    # Top right: the three strongest harmonics, stacked in the space of one panel.
    left, bottom, frame_width, frame_height = rects[1]
    gap = .012
    panel_height = (frame_height - 2*gap)/3
    map_axes = []
    for i, (mode, values) in enumerate(zip(modes, maps)):
        ax = fig.add_axes((left, bottom + (2-i)*(panel_height+gap), frame_width, panel_height))
        ax.pcolormesh(data.q, data.energy, values, cmap='RdBu_r', norm=mode_norm, shading='nearest', rasterized=True)
        ax.set_xlim(data.q[0], data.q[-1])
        ax.set_ylim(data.energy[0], data.energy[-1])
        style.style_axes(ax, heatmap=True, tick_size=font)
        ax.yaxis.set_major_locator(style.MaxNLocator(nbins=2))
        ax.xaxis.set_major_locator(style.MaxNLocator(nbins=4))
        ax.tick_params(labelbottom=i == 2)
        if i == 1:
            ax.set_ylabel(r'$E\ [\mathrm{eV}]$')
        if i == 2:
            ax.set_xlabel(r'$q\ [\mathrm{keV}]$')
        style.add_boxes(ax, [rf'$({mode[0]},{mode[1]})$'])
        map_axes.append(ax)
        l, m = mode
        ax_integrals.plot(data.q, data.integrals[:, l*l+l+m], label=rf'$({l},{m})$', **line_style(i))
    overview_colourbar(fig, map_axes[0], mode_norm, 'RdBu_r',
                       r'$S_{\ell m}(q,E)\ [\mathrm{\AA}^{-3}\,\mathrm{eV}^{-1}]$')
    # Bottom left: the same angular mean, integrated over energy.
    ax_total.plot(data.q, integrated_mean, **line_style(0))
    ax_total.set_ylim(bottom=0)
    ax_total.set_ylabel(r'$\int dE\,\langle S(\vec q,E)\rangle_\Omega\ [\mathrm{\AA}^{-3}]$')

    # Bottom right: the energy integrals of the same three harmonics, which do change sign.
    ax_integrals.axhline(0, color='.45', lw=.5)
    ax_integrals.set_ylabel(r'$\int dE\,S_{\ell m}(q,E)\ [\mathrm{\AA}^{-3}]$')
    ax_integrals.legend(loc='best', ncol=1, borderpad=.35)
    for ax in (ax_total, ax_integrals):
        style.style_axes(ax, tick_size=font)
        ax.set_xlim(data.q[0], data.q[-1])
        ax.set_xlabel(r'$q\ [\mathrm{keV}]$')
    return fig
