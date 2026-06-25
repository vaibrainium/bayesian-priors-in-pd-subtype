import matplotlib
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

BASE_DIR = Path(__file__).parent
STYLE = BASE_DIR / "dissemination.mplstyle"
matplotlib.style.use(STYLE)

# Constants
TEXTWIDTH = 9.0  # inches
FONTSIZE = 9.0
CMAP = cm.plasma
CMAP_R = cm.plasma_r
COLORS = [CMAP(i / 4.0) for i in range(5)]
COLORS_NEUTRAL = ["0.0", "0.4", "0.7", "1.0"]
STYLE_SETTINGS = ["dark_background", "presentation"]


# Apply default rcParams for consistent font sizes
def setup():
    """Apply general Matplotlib style and settings."""
    matplotlib.rcParams.update(
        {
            "font.size": FONTSIZE,
            "axes.titlesize": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "legend.fontsize": FONTSIZE,
            "figure.titlesize": FONTSIZE,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


# Helper function for margin calculation
def _calculate_margins(large_margin, small_margin, left_margin_large, right_margin_large, top_margin_large, bottom_margin_large):
    """Calculate margins based on input flags."""
    margins = {
        "left": large_margin if left_margin_large else small_margin,
        "right": large_margin if right_margin_large else small_margin,
        "top": large_margin if top_margin_large else small_margin,
        "bottom": large_margin if bottom_margin_large else small_margin,
    }
    return margins


# Helper function to calculate figure size based on margins
def _calculate_figure_size(width, height, margins, aspect_ratio=1.0):
    """Calculate figure size with margins applied."""
    left, right, top, bottom = margins.values()

    if width is not None:
        height = width / aspect_ratio
    elif height is not None:
        width = height * aspect_ratio

    return width, height


# Create figure by specifying height
def figure_by_height(
    height=TEXTWIDTH * 0.5,
    large_margin=0.14,
    small_margin=0.03,
    make3d=False,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """Create a figure, size specified by height."""
    margins = _calculate_margins(large_margin, small_margin, left_margin_large, right_margin_large, top_margin_large, bottom_margin_large)
    width, height = _calculate_figure_size(None, height, margins)

    fig = plt.figure(figsize=(width, height))
    ax = Axes3D(fig) if make3d else plt.gca()

    plt.subplots_adjust(
        left=margins["left"],
        right=1.0 - margins["right"],
        bottom=margins["bottom"],
        top=1.0 - margins["top"],
        wspace=0.0,
        hspace=0.0,
    )

    return fig, ax


# Create figure by specifying width
def figure_by_width(
    width=TEXTWIDTH * 0.5,
    large_margin=0.14,
    small_margin=0.03,
    make3d=False,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """Create a figure, size specified by width."""
    margins = _calculate_margins(large_margin, small_margin, left_margin_large, right_margin_large, top_margin_large, bottom_margin_large)
    width, height = _calculate_figure_size(width, None, margins)

    fig = plt.figure(figsize=(width, height))
    ax = Axes3D(fig) if make3d else plt.gca()

    plt.subplots_adjust(
        left=margins["left"],
        right=1.0 - margins["right"],
        bottom=margins["bottom"],
        top=1.0 - margins["top"],
        wspace=0.0,
        hspace=0.0,
    )

    return fig, ax


# Create figure with colorbar by specifying height
def figure_with_cbar_by_height(
    height=TEXTWIDTH * 0.5,
    large_margin=0.14,
    small_margin=0.03,
    cbar_sep=0.02,
    cbar_width=0.04,
    make3d=False,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """Create a figure with colorbar, size specified by height."""
    margins = _calculate_margins(large_margin, small_margin, left_margin_large, right_margin_large, top_margin_large, bottom_margin_large)
    width, height = _calculate_figure_size(None, height, margins)

    right = margins["right"] + cbar_width + cbar_sep
    cleft = 1.0 - (large_margin + cbar_width) * height / width
    cbottom = margins["bottom"]
    cwidth = cbar_width * height / width
    cheight = 1.0 - margins["top"] - margins["bottom"]

    fig = plt.figure(figsize=(width, height))
    ax = Axes3D(fig) if make3d else plt.gca()

    plt.subplots_adjust(
        left=margins["left"] * height / width,
        right=1.0 - right * height / width,
        bottom=margins["bottom"],
        top=1.0 - margins["top"],
        wspace=0.0,
        hspace=0.0,
    )
    cax = fig.add_axes([cleft, cbottom, cwidth, cheight])

    plt.sca(ax)

    return fig, (ax, cax)


# Grid of panels, no colorbars, size specified by height
def grid_by_height(
    nx=4,
    ny=2,
    height=0.5 * TEXTWIDTH,
    aspect_ratio=1.0,
    large_margin=0.14,
    small_margin=0.03,
    sep=0.02,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """Create a grid of panels with consistent margins."""
    margins = _calculate_margins(large_margin, small_margin, left_margin_large, right_margin_large, top_margin_large, bottom_margin_large)

    panel_size = (1.0 - margins["top"] - margins["bottom"] - (ny - 1) * sep) / ny
    width = height * aspect_ratio * (margins["left"] + nx * panel_size + (nx - 1) * sep + margins["right"])
    avg_width_abs = (height * panel_size * nx * ny) / (nx * ny + ny)
    avg_height_abs = height * panel_size
    wspace = sep * height / avg_width_abs
    hspace = sep * height / avg_height_abs

    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(ny, nx, width_ratios=[1.0] * nx, height_ratios=[1.0] * ny)
    plt.subplots_adjust(
        left=margins["left"] * height / width,
        right=1.0 - margins["right"] * height / width,
        bottom=margins["bottom"],
        top=1.0 - margins["top"],
        wspace=wspace,
        hspace=hspace,
    )
    return fig, gs


def grid_by_width(
    nx=4,
    ny=2,
    width=TEXTWIDTH,
    aspect_ratio=1.0,
    large_margin=0.14,
    small_margin=0.03,
    sep=0.02,
    left_margin_large=True,
    right_margin_large=False,
    bottom_margin_large=True,
    top_margin_large=False,
):
    """Grid of panels, no colorbars, size specified by width."""

    left = large_margin if left_margin_large else small_margin
    right = large_margin if right_margin_large else small_margin
    top = large_margin if top_margin_large else small_margin
    bottom = large_margin if bottom_margin_large else small_margin

    # Panel size calculation
    panel_size = (1.0 - top - bottom - (ny - 1) * sep) / ny
    height = width / (left + nx * panel_size + (nx - 1) * sep + right) / aspect_ratio

    # wspace and hspace calculation for optimal spacing
    avg_width_abs = (height * panel_size * nx * ny) / (nx * ny + ny)
    avg_height_abs = height * panel_size
    wspace = sep * height / avg_width_abs
    hspace = sep * height / avg_height_abs

    # Set up figure and adjust layout
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(ny, nx, width_ratios=[1.0] * nx, height_ratios=[1.0] * ny)
    plt.subplots_adjust(
        left=left * height / width,
        right=1.0 - right * height / width,
        bottom=bottom,
        top=1.0 - top,
        wspace=wspace,
        hspace=hspace,
    )
    return fig, gs


# Add scatter plot with consistent styling
def plot_scatter(xs, ys, **scatter_kwargs):
    """Plot a scatter plot with consistent styling."""
    defaults = {"alpha": 0.6, "lw": 3, "s": 80, "color": "C0", "facecolors": "none", "marker": "."}
    scatter_kwargs = {**defaults, **scatter_kwargs}
    plt.scatter(xs, ys, **scatter_kwargs)


# Plot a line with consistent styling
def plot_line(xs, ys, **plot_kwargs):
    """Plot a line with consistent styling."""
    plot_kwargs["linewidth"] = plot_kwargs.get("linewidth", 3)
    background_plot_kwargs = {**plot_kwargs, "linewidth": plot_kwargs["linewidth"] + 2, "color": "white"}
    del background_plot_kwargs["label"]

    plt.plot(xs, ys, **background_plot_kwargs, zorder=30)
    plt.plot(xs, ys, **plot_kwargs, zorder=31)


# Plot error bars (vertical)
def plot_errorbar(xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3):
    """Plot vertical error bars with consistent styling."""
    colors = [colors] * len(xs) if isinstance(colors, str) else colors
    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(x=x, y=y, yerr=np.array((err_l, err_u))[:, None], ls="none", color=colors[ii], zorder=1)
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)


# Plot error bars (horizontal)
def plot_x_errorbar(xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3):
    """Plot horizontal error bars with consistent styling."""
    colors = [colors] * len(xs) if isinstance(colors, str) else colors
    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(x=x, y=y, xerr=np.array((err_l, err_u))[:, None], ls="none", color=colors[ii], zorder=1)
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)


def add_significance_bracket(ax, x1, x2, y, p_value, bracket_height=None, fontsize=9, fmt="stars"):
    """Draw a significance bracket between two x positions.

    Parameters
    ----------
    fmt : str
        'stars' → *** / ** / * / ns  |  'p' → 'p = 0.032'  |  'p<' → 'p < 0.001'
    bracket_height : float or None
        Tick height. Defaults to 1 % of the current y-range.
    """
    ylim = ax.get_ylim()
    if bracket_height is None:
        bracket_height = (ylim[1] - ylim[0]) * 0.01

    ax.plot([x1, x1, x2, x2], [y, y + bracket_height, y + bracket_height, y], color="k", lw=1, clip_on=False)

    if fmt == "stars":
        if p_value < 0.001:
            label = "***"
        elif p_value < 0.01:
            label = "**"
        elif p_value < 0.05:
            label = "*"
        else:
            label = "ns"
    elif fmt == "p":
        label = f"p = {p_value:.3f}"
    else:
        label = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"

    ax.text((x1 + x2) / 2, y + bracket_height * 1.5, label, ha="center", va="bottom", fontsize=fontsize)


def bar_plot_with_significance(
    ax,
    data,
    sig_pairs=None,
    colors=None,
    hatches=None,
    bar_width=0.6,
    positions=None,
    correction="bonferroni",
    bracket_pad=0.05,
    bracket_fmt="stars",
    alpha=0.05,
    **bar_kwargs,
):
    """Bar plot with SEM error bars and optional significance brackets.

    Parameters
    ----------
    data : dict
        {label: array_of_values} — mean and SEM are computed from the array.
    sig_pairs : list of tuples, optional
        Each entry is either:
          (label1, label2)           — Wilcoxon is run automatically; Bonferroni
                                       correction applied across all auto pairs.
          (label1, label2, p_value)  — use pre-computed p_value; no correction.
    colors : dict or str, optional
        {label: color} mapping, or a single color applied to all bars.
    hatches : dict, optional
        {label: hatch} mapping.
    positions : list, optional
        x positions for each bar. Defaults to 0, 1, 2, …
    correction : str
        multipletests method used when sig_pairs are auto-computed.
    bracket_pad : float
        Extra vertical gap between the tallest bar+SEM and the first bracket.
    bracket_fmt : str
        Passed to add_significance_bracket — 'stars', 'p', or 'p<'.
    alpha : float
        Significance level (reserved for future use).
    **bar_kwargs
        Forwarded to ax.bar (e.g. edgecolor, linewidth).

    Returns
    -------
    dict mapping each label to its x position.
    """
    labels = list(data.keys())
    if positions is None:
        positions = list(range(len(labels)))
    pos_map = dict(zip(labels, positions))

    bar_kwargs.setdefault("edgecolor", "k")
    bar_kwargs.setdefault("linewidth", 1)

    tops = {}
    for label, pos in pos_map.items():
        vals = np.asarray(data[label])
        mean = np.nanmean(vals)
        sem = scipy_stats.sem(vals, nan_policy="omit")

        color = colors.get(label, "steelblue") if isinstance(colors, dict) else (colors or "steelblue")
        hatch = hatches.get(label, "") if hatches else ""

        ax.bar(pos, mean, width=bar_width, color=color, hatch=hatch, **bar_kwargs)
        ax.errorbar(pos, mean, yerr=sem, fmt="none", color="k", capsize=6, elinewidth=2, capthick=2)
        tops[label] = mean + sem

    if sig_pairs:
        auto_pairs = [t for t in sig_pairs if len(t) == 2]
        manual_pairs = [t for t in sig_pairs if len(t) == 3]

        computed_p = {}
        if auto_pairs:
            p_vals = [wilcoxon(data[a], data[b]).pvalue for a, b in auto_pairs]
            corrected = multipletests(p_vals, method=correction)[1]
            computed_p = {(a, b): cp for (a, b), cp in zip(auto_pairs, corrected)}

        all_pairs = [(a, b, computed_p[(a, b)]) for a, b in auto_pairs] + list(manual_pairs)

        bracket_y = max(tops.values()) + bracket_pad
        bracket_step = bracket_pad * 1.8

        for a, b, p in all_pairs:
            add_significance_bracket(ax, pos_map[a], pos_map[b], bracket_y, p, fmt=bracket_fmt)
            bracket_y += bracket_step

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    return pos_map
