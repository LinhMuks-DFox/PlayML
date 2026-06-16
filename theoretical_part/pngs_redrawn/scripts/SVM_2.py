"""Redraw of SVM_2: SVM motivation — a correct but non-optimal decision boundary.

Concept diagram: two linearly separable 2D clusters (blue lower-left, red/orange-red
upper-right) with a single blue straight line (negative slope) that correctly separates
the two classes but is deliberately drawn hugging the red cluster (close to red, far
from blue).

The narrative point: a boundary that classifies training data correctly is not unique;
one that crowds one class generalizes poorly. This sets up the max-margin SVM concept.
No model is trained — points and line are synthetic to keep the illustration clear.

Style: unified _style.py, large dots (s=200), no tick numbers, no grid, no legend,
no title, arrow-tipped bottom/left axes, blue decision boundary line.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_2.png"

# Colors matching the spec: blue cluster, red/orange-red cluster, blue boundary.
BLUE_COLOR = _style.PALETTE[0]   # #2563EB  (blue)
RED_COLOR = _style.PALETTE[1]    # #EF4444  (red)
LINE_COLOR = "royalblue"         # blue decision boundary per spec

# Coordinate bounds (shared with SVM_3/SVM_4 series).
X_MIN, X_MAX = 0.0, 10.0
Y_MIN, Y_MAX = 0.0, 8.0


def build_clusters():
    """Synthetic 2D points: linearly separable with a gap between clusters.

    Blue cluster: lower-left region.
    Red cluster: upper-right region.
    The gap is large enough that a decision boundary can be drawn off-center,
    deliberately biased (crowding) toward the red cluster.
    """
    # Blue cluster (lower-left), ~8 points.
    blue = np.array(
        [
            [1.2, 1.8],
            [2.0, 3.1],
            [1.5, 4.2],
            [3.0, 2.0],
            [3.5, 3.4],
            [2.8, 4.8],
            [4.5, 2.8],
            [4.0, 4.0],
        ]
    )
    # Red cluster (upper-right), ~7 points.
    red = np.array(
        [
            [5.2, 6.8],
            [6.5, 5.9],
            [7.8, 7.1],
            [6.0, 5.0],
            [8.3, 5.8],
            [7.2, 4.4],
            [9.0, 6.3],
        ]
    )
    return blue, red


def main():
    blue, red = build_clusters()

    fig, ax = _style.new_ax(figsize=(6, 5))

    # --- Decision boundary ---------------------------------------------------
    # Line: y = slope * x + intercept
    # Deliberately placed close to the red cluster (hugging it from below)
    # to illustrate the ill-posed / poor-generalization problem.
    # At x=0: y ~ 7.2; at x=10: y ~ 3.7  → diagonal from upper-left to lower-right.
    slope = -0.35
    intercept = 7.0
    x_line = np.array([X_MIN, X_MAX])
    y_line = slope * x_line + intercept
    ax.plot(
        x_line,
        y_line,
        color=LINE_COLOR,
        linewidth=2.4,
        zorder=2,
        solid_capstyle="round",
    )

    # --- Scatter plots -------------------------------------------------------
    ax.scatter(
        blue[:, 0],
        blue[:, 1],
        s=180,
        c=BLUE_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
    )
    ax.scatter(
        red[:, 0],
        red[:, 1],
        s=180,
        c=RED_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
    )

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    # No grid, no tick numbers, no labels, no title.
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Hide all default spines; replace with arrow-tipped axes from origin.
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    arrow_kw = dict(
        arrowstyle="-|>",
        color="#333333",
        linewidth=1.6,
        mutation_scale=20,
    )
    # x-axis arrow: origin → right edge.
    ax.annotate("", xy=(X_MAX, 0), xytext=(0, 0),
                xycoords="data", textcoords="data", arrowprops=arrow_kw)
    # y-axis arrow: origin → top edge.
    ax.annotate("", xy=(0, Y_MAX), xytext=(0, 0),
                xycoords="data", textcoords="data", arrowprops=arrow_kw)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
