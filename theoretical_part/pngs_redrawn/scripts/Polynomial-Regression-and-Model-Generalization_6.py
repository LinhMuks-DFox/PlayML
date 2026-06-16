"""Bias-Variance trade-off illustrated as a 2x2 grid of dartboard targets.

Columns: Low / High Variance.  Rows: Low / High Bias.
In each cell a cluster of shot marks (model predictions) shows how far from
the red center (bias) and how scattered (variance) the predictions land.

Original re-drawing for the ML notes figure series. Concentric circles are
matplotlib.patches.Circle; shot marks are synthetic numpy Gaussian samples
with a fixed seed for reproducibility.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
from matplotlib.patches import Circle

import _style  # noqa: F401  (applies unified rcParams on import)

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "Polynomial-Regression-and-Model-Generalization_6.png"
)

# --- Target appearance -----------------------------------------------------
# Concentric rings drawn large -> small so the inner ones overpaint the outer.
RING_RADII = [1.00, 0.66, 0.34]
RING_FACES = ["white", "#9ECBEC", "#2F7DC4"]  # outer white, mid light-blue, inner deep-blue
RING_EDGE = "#222222"
CENTER_RADIUS = 0.14
CENTER_COLOR = "#E03131"  # red = true target

SHOT_COLOR = "#1E3A8A"  # deep blue shot marks
SHOT_SIZE = 26

# --- Per-cell bias / variance configuration --------------------------------
# bias  -> (dx, dy) offset of the shot-cluster mean from the bullseye center
# std   -> scatter (variance) of the shots around that mean
CELLS = {
    # (row, col): label is implicit from grid position
    (0, 0): dict(bias=(0.0, 0.0), std=0.045),   # Low Bias,  Low Variance  (ideal)
    (0, 1): dict(bias=(0.0, 0.0), std=0.34),    # Low Bias,  High Variance
    (1, 0): dict(bias=(-0.32, 0.30), std=0.05), # High Bias, Low Variance
    (1, 1): dict(bias=(-0.10, 0.46), std=0.30), # High Bias, High Variance
}

N_SHOTS = 22
RNG_SEED = 7

COL_TITLES = ["Low Variance", "High Variance"]
ROW_TITLES = ["Low Bias", "High Bias"]


def draw_target(ax):
    """Draw the concentric-ring bullseye on a clean square axis."""
    for r, fc in zip(RING_RADII, RING_FACES):
        ax.add_patch(
            Circle((0, 0), r, facecolor=fc, edgecolor=RING_EDGE,
                   linewidth=1.6, zorder=1)
        )
    # Red center = true target.
    ax.add_patch(
        Circle((0, 0), CENTER_RADIUS, facecolor=CENTER_COLOR,
               edgecolor="none", zorder=2)
    )
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_shots(ax, bias, std, rng):
    """Scatter the Gaussian shot marks, clipped to stay on the board."""
    dx, dy = bias
    xs = rng.normal(dx, std, size=N_SHOTS)
    ys = rng.normal(dy, std, size=N_SHOTS)
    # Keep points inside the outer ring (radius ~1.0) to avoid stray overflow.
    rad = np.hypot(xs, ys)
    scale = np.where(rad > 1.05, 1.05 / rad, 1.0)
    xs, ys = xs * scale, ys * scale
    ax.scatter(xs, ys, s=SHOT_SIZE, color=SHOT_COLOR,
               edgecolor="white", linewidth=0.4, zorder=3)


def main():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.4))
    rng = np.random.default_rng(RNG_SEED)

    for (r, c), cfg in CELLS.items():
        ax = axes[r, c]
        draw_target(ax)
        draw_shots(ax, cfg["bias"], cfg["std"], rng)

    # Column titles (top of each top-row subplot).
    for c, title in enumerate(COL_TITLES):
        axes[0, c].set_title(title, fontsize=15, pad=14)

    # Row titles (rotated, to the left of each left-column subplot).
    for r, title in enumerate(ROW_TITLES):
        axes[r, 0].text(
            -1.42, 0.0, title, rotation=90, ha="center", va="center",
            fontsize=15, transform=axes[r, 0].transData,
        )

    fig.suptitle("偏差-方差权衡 (Bias-Variance Trade-off)", fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.03,
                        wspace=0.12, hspace=0.18)

    # finalize() toggles spines (harmless on axis('off')) and saves the PNG.
    _style.finalize(fig, OUT_PATH)


if __name__ == "__main__":
    main()
