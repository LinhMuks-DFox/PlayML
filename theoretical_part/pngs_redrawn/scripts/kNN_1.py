"""Redraw of kNN_1: the kNN input dataset scatter plot.

Two-class 2D scatter of tumor samples. Horizontal axis = tumor size,
vertical axis = (growth) time. Red = benign, blue = malignant. This is
the first kNN concept figure: only the raw training data, no query point
yet. Style matches the unified _style.py (CJK-capable fonts, clean look),
while keeping the original hand-drawn PPT feel: large dots, no tick
numbers, arrow-tipped left/bottom axes.

Later figures (kNN_2/3/4) add a green query point + nearest-neighbor
links on top of this same base layout.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/kNN_1.png"

# Colors mapped to the notes text: red = benign, blue = malignant.
# (Reuse the unified palette entries instead of raw hex.)
BENIGN_COLOR = _style.PALETTE[1]     # red  #EF4444
MALIGNANT_COLOR = _style.PALETTE[0]  # blue #2563EB


def build_dataset():
    """Hand-placed synthetic 2D points: roughly separable but overlapping.

    Red (benign) cluster sits lower-left, blue (malignant) cluster sits
    upper-right, with a couple of crossing points to convey real
    classification difficulty.
    """
    # Benign (label 0): lower-left, plus one point pushed toward the middle.
    benign = np.array(
        [
            [1.0, 1.7],
            [1.1, 0.6],
            [2.3, 1.2],
            [4.3, 2.0],  # the intruding red point near the blue side
        ]
    )
    # Malignant (label 1): upper-right, plus one lower point overlapping reds.
    malignant = np.array(
        [
            [4.0, 4.5],
            [6.6, 5.2],
            [7.5, 4.0],
            [5.6, 2.9],  # the lower blue point near the red side
        ]
    )
    return benign, malignant


def main():
    benign, malignant = build_dataset()

    fig, ax = _style.new_ax(figsize=(7, 4.6))

    # Large filled circles, white edge for a clean PPT-style pop.
    ax.scatter(
        benign[:, 0],
        benign[:, 1],
        s=520,
        c=BENIGN_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
        label="良性",
    )
    ax.scatter(
        malignant[:, 0],
        malignant[:, 1],
        s=520,
        c=MALIGNANT_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
        label="恶性",
    )

    # Plot bounds with padding so arrows + dots have breathing room.
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 6.2)

    # Hand-drawn PPT feel: no grid, no tick numbers.
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Arrow-tipped left/bottom axes; hide top/right (finalize also hides them).
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    arrow_kw = dict(
        arrowstyle="-|>",
        color="#333333",
        linewidth=1.8,
        mutation_scale=22,
    )
    # Bottom axis arrow (x): from origin to the right edge.
    ax.annotate("", xy=(8.6, 0), xytext=(0, 0), arrowprops=arrow_kw)
    # Left axis arrow (y): from origin to the top edge.
    ax.annotate("", xy=(0, 6.2), xytext=(0, 0), arrowprops=arrow_kw)

    # Axis labels placed near the arrow heads, matching the original layout.
    ax.text(
        8.55,
        -0.25,
        "肿瘤大小",
        ha="right",
        va="top",
        fontsize=13,
    )
    ax.text(
        -0.15,
        6.15,
        "时间",
        ha="right",
        va="top",
        fontsize=13,
    )

    # Title in red, like the original.
    ax.set_title("k近邻算法", color=BENIGN_COLOR, fontsize=20, pad=14)

    ax.legend(loc="lower right", framealpha=0.9)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
