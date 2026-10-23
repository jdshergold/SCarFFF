"""Computer Modern styles and shared print layouts for SCarFFF figures."""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np

PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#56B4E9']
LINESTYLES = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]
CAPTION_BOX = dict(boxstyle='round,pad=0.34,rounding_size=0.08', facecolor='#f7f7f7',
                   edgecolor='#8a8a8a', linewidth=0.6, alpha=0.95)



def use_style(caption_pt=9):
    """
    Set the Computer Modern font and house plot style.

    # Arguments:
    - caption_pt::float: The final caption font size, in TeX points.
    """
    mpl.rcParams.update({'text.usetex': True, 'font.family': 'serif',
                         'font.serif': ['Computer Modern'], 'font.size': caption_pt,
                         'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
                         'axes.labelsize': caption_pt, 'axes.linewidth': 0.65,
                         'xtick.labelsize': caption_pt, 'ytick.labelsize': caption_pt,
                         'legend.fontsize': caption_pt, 'legend.frameon': True,
                         'legend.framealpha': 0.9, 'legend.fancybox': False,
                         'legend.edgecolor': '0.6'})


def style_axes(ax, heatmap=False, tick_size=9):
    """
    Set the ticks, frame and grid for one plot.

    # Arguments:
    - ax::matplotlib.axes.Axes: The plot axes.
    - heatmap::bool: Whether the plot is a heatmap.
    - tick_size::int: The tick label size in points.
    """
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='both', direction='out', top=False, right=False, labelsize=tick_size)
    ax.tick_params(which='major', length=3, width=0.6)
    ax.tick_params(which='minor', length=1.5, width=0.5)
    ax.spines['top'].set_alpha(0.45)
    ax.spines['right'].set_alpha(0.45)
    if heatmap:
        ax.grid(False, which='both')
    else:
        ax.grid(which='major', color='#cfcfcf', linestyle='--', linewidth=0.6, alpha=0.8)
        ax.grid(which='minor', color='#e7e7e7', linestyle=':', linewidth=0.45, alpha=0.7)



@dataclass
class PrintSettings:
    """The final printed size and requested figures at one width."""

    width: str = 'full'
    shape: str = 'rectangular'
    caption_pt: float = 9
    text_width_pt: float = 510
    gap_pt: float = 18
    output_dir: Path = Path('.')
    file_type: tuple = ('png', 'pdf')
    figures: list = field(default_factory=list)


def make_frame(args, square=False):
    """
    Make a plot at a specified printed width, before measuring its final margins.

    # Arguments:
    - args::PrintSettings: The width and caption-size settings.
    - square::bool: Whether the plotting frame must be square.

    # Returns:
    - tuple: The figure and its plotting axes.
    """
    # Share the text width between however many figures sit side by side, minus the gaps between
    # them. The 72.27 takes points to inches.
    columns = {'full': 1, 'half': 2, 'third': 3}[args.width]
    width = (args.text_width_pt - (columns - 1) * args.gap_pt) / columns / 72.27

    # Margins in inches, leaving room for the labels and the colourbar above.
    left, right, bottom, top = .70, .14, .43, .66
    if args.width == 'third':
        top = .83
    frame_width = width - left - right
    if frame_width < 1:
        raise ValueError('The requested width leaves less than one inch for the plot frame.')

    # The frame itself, square when asked for and otherwise flatter at larger widths.
    shape = args.shape or ('square' if args.width == 'third' else 'rectangular')
    height_ratio = .42 if args.width == 'full' else .62
    frame_height = frame_width if square or shape == 'square' else height_ratio * frame_width
    height = bottom + frame_height + top
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((left / width, bottom / height,
                       frame_width / width, frame_height / height))
    style_axes(ax, tick_size=args.caption_pt)

    # Fewer ticks on the narrower figures, so the labels do not collide.
    if args.width == 'third':
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    else:
        bins = 10 if args.width == 'full' else 5
        ax.xaxis.set_major_locator(MaxNLocator(nbins=bins, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=bins, steps=[1, 2, 2.5, 5, 10]))
    return fig, ax


def add_colourbar(fig, ax, norm, cmap, label, args, signed=False):
    """
    Put a labelled colourbar immediately above the plot frame.

    # Arguments:
    - fig::matplotlib.figure.Figure: The figure.
    - ax::matplotlib.axes.Axes: The plot axes.
    - norm::matplotlib.colors.Normalize: The colour scale.
    - cmap::str: The colour map.
    - label::str: The quantity and its units, if any.
    - args::PrintSettings: The printed width setting.
    - signed::bool: Whether to mark the signed logarithmic colour scale.
    """
    # Span the bar across every panel, ignoring any colourbar already placed.
    axes = [axes for axes in fig.axes if axes.get_label() != 'colourbar']
    position = Bbox.union([axes.get_position() for axes in axes])
    height = fig.get_figheight()
    cax = fig.add_axes((position.x0, position.y1 + .065 / height,
                        position.width, .065 / height), label='colourbar')

    # Choose the ticks to suit the scale and the width: decades on a log scale, decades either side
    # of zero on a signed one, and otherwise a round set of linear ticks.
    logarithmic = isinstance(norm, LogNorm)
    if logarithmic:
        powers = np.arange(np.ceil(np.log10(norm.vmin)), np.floor(np.log10(norm.vmax)) + 1)
        ticks = 10.0**powers
        if args.width == 'third':
            ticks = ticks[::2]
    elif args.width == 'third':
        ticks = [-norm.vmax, 0, norm.vmax] if signed else np.linspace(0, norm.vmax, 3)
    elif signed and isinstance(norm, SymLogNorm):
        highest_power = int(np.floor(np.log10(norm.vmax)))
        powers = np.arange(highest_power - 3, highest_power + 1)
        positive_ticks = 10.0**powers
        if args.width == 'half':
            positive_ticks = positive_ticks[1::2]
        ticks = np.concatenate([-positive_ticks[::-1], [0], positive_ticks])
    else:
        lower = norm.vmin if signed else 0
        ticks = MaxNLocator(nbins=8 if args.width == 'full' else 7,
                            steps=[1, 2, 2.5, 5, 10]).tick_values(lower, norm.vmax)
        ticks = ticks[(ticks >= lower) & (ticks <= norm.vmax)]
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation='horizontal', ticks=ticks)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.set_label(label, labelpad=6)
    cb.ax.tick_params(length=2.5, pad=2)
    cb.outline.set_linewidth(.6)
    # Powers of ten on the logarithmic scales, plain numbers otherwise.
    if logarithmic or (signed and isinstance(norm, SymLogNorm) and args.width != 'third'):
        cb.formatter = FuncFormatter(
            lambda value, position: '$0$' if value == 0 else
            rf'${"-" if value < 0 else ""}10^{{{int(round(np.log10(abs(value))))}}}$')
    else:
        cb.formatter = FuncFormatter(lambda value, position: f'{value:.2g}')
    cb.update_ticks()

    # Pull the end labels inwards, so they do not hang off the ends of the bar.
    labels = cb.ax.get_xticklabels()
    if args.width == 'third' or not signed:
        if np.isclose(ticks[0], norm.vmin, rtol=1e-10, atol=0):
            labels[0].set_ha('left')
        if np.isclose(ticks[-1], norm.vmax, rtol=1e-10, atol=0):
            labels[-1].set_ha('right')


def add_line_legend(ax, args, title):
    """
    Use the same legend layout for modes and transitions at each printed width.

    # Arguments:
    - ax::matplotlib.axes.Axes: The curve axes.
    - args::PrintSettings: The width and caption font size.
    - title::str: The legend heading at full and half width.
    """
    # At third width there is no room inside the frame, so the legend goes above it and loses its
    # title.
    columns = {'full': 4, 'half': 2, 'third': 2}[args.width]
    position = dict(loc='lower center', bbox_to_anchor=(.5, 1.04)) if args.width == 'third' else dict(loc='best')
    legend = ax.legend(**position, title=None if args.width == 'third' else title,
                       title_fontsize=args.caption_pt, ncol=columns, columnspacing=.6,
                       handlelength=1.1 if args.width == 'third' else 1.5, borderpad=.35)
    legend.get_frame().set_linewidth(.5)


def place_frames(fig, rectangle):
    """
    Place one frame, two stacked frames, or a grid of six within a shared outer area.

    # Arguments:
    - fig::matplotlib.figure.Figure: The figure.
    - rectangle::tuple: The outer left, bottom, width and height, in inches.
    """
    # Six panels go in two columns of three, anything else in a single column.
    left, bottom, frame_width, frame_height = rectangle
    width, height = fig.get_size_inches()
    axes = [ax for ax in fig.axes if ax.get_label() != 'colourbar']
    columns = 2 if len(axes) == 6 else 1
    rows = len(axes) // columns
    gap_x = .32 if columns > 1 else 0
    gap_y = .09 if rows > 1 else 0
    panel_width = (frame_width - (columns - 1) * gap_x) / columns
    panel_height = (frame_height - (rows - 1) * gap_y) / rows

    # Fill the grid top to bottom, then left to right.
    for i, ax in enumerate(axes):
        row, column = i % rows, i // rows
        panel_bottom = bottom + (rows - 1 - row) * (panel_height + gap_y)
        panel_left = left + column * (panel_width + gap_x)
        ax.set_position((panel_left / width, panel_bottom / height,
                         panel_width / width, panel_height / height))
        if ax.texts:
            align_boxes(ax, list(ax.texts))
    # With a stack of panels, centre the one y label against the whole stack rather than its panel.
    if rows > 1:
        centre = (bottom + frame_height/2) / height
        for ax in axes:
            if ax.get_ylabel():
                position = ax.get_position()
                ax.yaxis.label.set_y((centre - position.y0) / position.height)

    # Stretch any colourbar back across the full width the panels now occupy.
    for cax in [ax for ax in fig.axes if ax.get_label() == 'colourbar']:
        cax.set_position((left / width, (bottom + frame_height + .065) / height,
                          frame_width / width, .065 / height))


def add_boxes(ax, labels):
    """
    Stack annotations with aligned right edges.

    # Arguments:
    - ax::matplotlib.axes.Axes: The plot axes.
    - labels::list: The annotation strings.
    """
    boxes = []
    for label, colour in zip(labels, ['#eef3f7', '#f8f1e8']):
        box = ax.text(.95, .95, label, transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(CAPTION_BOX, facecolor=colour), zorder=40)
        boxes.append(box)
    align_boxes(ax, boxes)


def align_boxes(ax, boxes):
    """
    Align the annotation boxes after setting the plot size.

    # Arguments:
    - ax::matplotlib.axes.Axes: The plot axes.
    - boxes::list: The annotation text objects.
    """
    # The box size is only known once it has been drawn, so draw, measure, then nudge it into place
    # and work down the stack.
    top = .95
    for box in boxes:
        ax.figure.canvas.draw()
        bounds = box.get_bbox_patch().get_window_extent(
            ax.figure.canvas.get_renderer()).transformed(ax.transAxes.inverted())
        x, y = box.get_position()
        box.set_position((x + .95 - bounds.x1, y + top - bounds.y1))
        top -= bounds.height + .03


def save_plot(fig, args, name, square=False):
    """
    Keep a finished plot for measuring margins with the other plots of its shape.

    # Arguments:
    - fig::matplotlib.figure.Figure: The figure.
    - args::PrintSettings: The output directory and width setting.
    - name::str: The quantity and selection in the filename.
    - square::bool: Whether the quantity requires a square frame.
    """
    shape = 'square' if square else args.shape or ('square' if args.width == 'third' else 'rectangular')

    # Number a repeated name rather than overwriting the earlier figure.
    original_name = name
    duplicate = 2
    while any(saved_name == name and saved_shape == shape for _, saved_name, saved_shape in args.figures):
        name = f'{original_name}_{duplicate}'
        duplicate += 1

    # Hold the figure back, since the margins are only settled once every figure of this shape is in.
    args.figures.append((fig, name, shape))
    plt.close(fig)


def save_figures(args):
    """
    Measure shared margins for each frame shape and save the selected figures.

    # Arguments:
    - args::PrintSettings: The figures, printed width and output directory.
    """
    for shape in ('square', 'rectangular'):
        figures = [(fig, name) for fig, name, frame_shape in args.figures if frame_shape == shape]
        if not figures:
            continue
        width = figures[0][0].get_figwidth()
        ratio = 1 if shape == 'square' else .42 if args.width == 'full' else .62
        # Leave a small gap beyond the outermost label or legend.
        padding = 2.5 / 72.27
        # Measuring the margins moves the frames, which moves the labels, so repeat until it settles.
        for iteration in range(5):
            # Take the largest margin any of these figures needs, so they all end up the same size.
            margins = []
            for fig, name in figures:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                bounds = fig.get_tightbbox(renderer)
                axes = [ax for ax in fig.axes if ax.get_label() != 'colourbar']
                frame = Bbox.union([ax.get_window_extent(renderer) for ax in axes])
                frame = frame.transformed(fig.dpi_scale_trans.inverted())
                margins.append((frame.x0 - bounds.x0, bounds.x1 - frame.x1,
                                frame.y0 - bounds.y0, bounds.y1 - frame.y1))
            left, right, bottom, top = np.max(margins, axis=0) + padding
            frame_width = width - left - right
            if frame_width < 1:
                raise ValueError('The labels leave less than one inch for the plot frame.')
            frame_height = ratio * frame_width
            height = bottom + frame_height + top
            for fig, name in figures:
                fig.set_size_inches(width, height)
                place_frames(fig, (left, bottom, frame_width, frame_height))

        # Now every figure of this shape has the same frame, so write them out.
        for fig, name in figures:
            for extension in args.file_type:
                fig.savefig(args.output_dir / f'{name}_{args.width}_{shape}.{extension}', dpi=220)
            plt.close(fig)

