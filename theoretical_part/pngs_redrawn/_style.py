"""Shared "unified modern" matplotlib style for ML notes figures.

Import this module in any plotting script and use ``new_ax`` / ``finalize``
to get a consistent, clean, modern look across all generated PNGs.
"""

import matplotlib

# Non-interactive backend: safe for headless / batch figure generation.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from cycler import cycler

# --- Unified palette -------------------------------------------------------
# 8 modern, high-contrast colors (clear separation, works on white).
PALETTE = [
    "#2563EB",  # blue
    "#EF4444",  # red
    "#10B981",  # green
    "#F59E0B",  # amber
    "#8B5CF6",  # violet
    "#06B6D4",  # cyan
    "#EC4899",  # pink
    "#64748B",  # slate gray
]

# CJK-capable sans-serif font stack (first available wins). Generic
# 'sans-serif' kept last as a final fallback for non-CJK environments.
_FONT_STACK = [
    "PingFang HK",
    "Hiragino Sans GB",
    "Heiti TC",
    "Arial Unicode MS",
    "DejaVu Sans",
    "sans-serif",
]


def _apply_rcparams():
    """Apply the unified rcParams for a clean, modern look."""
    matplotlib.rcParams.update(
        {
            # Fonts (sans-serif + CJK fallback so Chinese renders correctly).
            "font.family": "sans-serif",
            "font.sans-serif": _FONT_STACK,
            "axes.unicode_minus": False,
            # Font sizes.
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            # Clean white background.
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            # Light gray thin grid (low alpha).
            "axes.grid": True,
            "grid.color": "#B0B0B0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,
            # Lines.
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            # Default figure size and high-quality saving.
            "figure.figsize": (6, 4),
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            # Legend styling.
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#CCCCCC",
            "legend.fancybox": True,
            # Spines / ticks.
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Default color cycle from PALETTE.
            "axes.prop_cycle": cycler(color=PALETTE),
        }
    )


# Apply the style at import time.
_apply_rcparams()


def new_ax(figsize=None):
    """Create a styled ``(fig, ax)`` pair.

    Parameters
    ----------
    figsize : tuple, optional
        Figure size in inches. Defaults to the unified ``(6, 4)``.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if figsize is None:
        figsize = matplotlib.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def finalize(fig, out_path):
    """Finalize and save a figure as PNG.

    Removes the top/right spines on every axes, runs ``tight_layout``,
    saves to ``out_path``, and closes the figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to finalize.
    out_path : str
        Destination PNG path.
    """
    for ax in fig.get_axes():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)
    return out_path
