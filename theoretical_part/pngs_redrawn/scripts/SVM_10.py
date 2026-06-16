"""Redraw of SVM_10: two-class scatter with a single blue outlier.

Concept diagram that motivates the Soft Margin SVM. Two linearly separable
2D clusters are drawn -- red in the upper-right, blue in the lower-left --
but one blue point is deliberately placed in the lower-right, *inside* the
red cluster's region, as an outlier / noise sample.

The figure draws ONLY the axes and the sample points: no decision boundary,
no margins. The point of this first figure in the SVM_10/11/12 series is to
show the data distribution + the lone blue outlier; later figures overlay
different boundaries on the *same* point coordinates. A Hard Margin SVM on
this distribution would be dragged strongly toward that outlier and overfit,
which is exactly the motivation for the soft margin.

Style matches the unified _style.py while keeping the original hand-drawn
PPT feel: large dots, no tick numbers, arrow-tipped left/bottom axes, no
grid, no title. The point coordinates are hand-fixed (not random-fit) so the
SVM_10/11/12 series stays on one consistent coordinate frame.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_10.png"

# Colors: red cluster (class A / +1), blue cluster (class B / -1).
RED_COLOR = _style.PALETTE[1]   # red  #EF4444
BLUE_COLOR = _style.PALETTE[0]  # blue #2563EB
TEXT_COLOR = "#1F2937"          # near-black for the annotation text

# Plot bounds (shared frame with the SVM_2/3/4 series for consistency).
X_MIN, X_MAX = 0.0, 10.0
Y_MIN, Y_MAX = 0.0, 7.6


def build_dataset():
    """Hand-placed synthetic 2D points (fixed -> reproducible, reusable).

    - Red cluster (class +1): upper-right, ~7 points.
    - Blue cluster (class -1): lower-left, ~6 points.
    - One blue OUTLIER placed in the lower-right, mixed into the red region.

    Returned coordinates are intended to be reused verbatim by SVM_11/SVM_12
    so those figures can overlay different decision boundaries on the same
    distribution.
    """
    # Red (upper-right), 7 points.
    red = np.array(
        [
            [4.6, 6.9],
            [6.6, 6.2],
            [8.2, 6.7],
            [5.7, 5.4],
            [8.2, 5.1],
            [7.1, 4.0],
            [6.4, 3.3],
        ]
    )
    # Blue main cluster (lower-left), 6 points.
    blue = np.array(
        [
            [2.3, 3.6],
            [1.3, 2.2],
            [2.6, 2.5],
            [4.0, 2.7],
            [4.2, 1.4],
            [3.0, 1.5],
        ]
    )
    # Single blue OUTLIER, placed lower-right, deep inside the red region.
    outlier = np.array([8.4, 2.2])
    return red, blue, outlier


def main():
    red, blue, outlier = build_dataset()

    fig, ax = _style.new_ax(figsize=(7, 5))

    # --- scatter: large filled circles, white edge for the clean PPT pop ----
    # Red cluster (class A / +1), upper-right.
    ax.scatter(
        red[:, 0],
        red[:, 1],
        s=200,
        c=RED_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
    )
    # Blue main cluster (class B / -1), lower-left.
    ax.scatter(
        blue[:, 0],
        blue[:, 1],
        s=200,
        c=BLUE_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
    )
    # The lone blue outlier in the red region (lower-right).
    ax.scatter(
        outlier[0],
        outlier[1],
        s=200,
        c=BLUE_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
    )

    # --- annotate the outlier as 异常点 / 噪音 -------------------------------
    # A short curved arrow points to the outlier from upper-right white space.
    ax.annotate(
        "异常点 / 噪音",
        xy=(outlier[0], outlier[1]),
        xytext=(outlier[0] - 0.2, outlier[1] - 1.35),
        color=TEXT_COLOR,
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="-|>",
            color=TEXT_COLOR,
            linewidth=1.5,
            shrinkA=2,
            shrinkB=8,
            connectionstyle="arc3,rad=0.25",
        ),
        zorder=4,
    )

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    # Hand-drawn PPT feel: no grid, no tick numbers.
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Arrow-tipped left/bottom axes; hide all default spines.
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    arrow_kw = dict(
        arrowstyle="-|>",
        color="#333333",
        linewidth=1.8,
        mutation_scale=22,
    )
    # Bottom axis arrow (x): origin -> right edge.
    ax.annotate("", xy=(X_MAX, 0), xytext=(0, 0), arrowprops=arrow_kw)
    # Left axis arrow (y): origin -> top edge.
    ax.annotate("", xy=(0, Y_MAX), xytext=(0, 0), arrowprops=arrow_kw)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
