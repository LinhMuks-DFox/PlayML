"""Redraw of SVM_1: the "decision boundary is not unique" concept diagram.

A conceptual sketch motivating SVM. Two well-separated classes of 2D points
(red cluster in the upper-right, blue cluster in the lower-left) are drawn,
together with two different straight lines A and B. Both lines correctly
separate the two classes, illustrating that for a linearly separable problem
the separating boundary is *not unique* (the ill-posed problem faced by plain
logistic regression). This motivates SVM's search for the "optimal" boundary.

The two lines have similar negative slopes but different intercepts, so they
are nearly parallel and both lie in the gap between the two clusters -- exactly
as in the original hand-drawn diagram.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_1.png"

RED = _style.PALETTE[1]    # #EF4444  -- class "upper right"
BLUE = _style.PALETTE[0]   # #2563EB  -- class "lower left"
LINE_COLOR = "#3B2DD0"     # deep blue/violet for both boundary lines
TEXT_COLOR = "#1F2937"     # near-black for the A / B text labels


def main():
    rng = np.random.default_rng(42)

    # Blue cluster: lower-left region, ~9 points
    blue_x = rng.uniform(0.5, 3.5, 9)
    blue_y = rng.uniform(0.5, 3.0, 9)

    # Red cluster: upper-right region, ~9 points
    red_x = rng.uniform(5.0, 8.5, 9)
    red_y = rng.uniform(4.5, 7.5, 9)

    fig, ax = _style.new_ax(figsize=(6, 5))

    # --- scatter the two classes -------------------------------------------
    ax.scatter(blue_x, blue_y, color=BLUE, s=160, edgecolors="white",
               linewidths=1.2, zorder=3)
    ax.scatter(red_x, red_y, color=RED, s=160, edgecolors="white",
               linewidths=1.2, zorder=3)

    # --- view box -----------------------------------------------------------
    x_lo, x_hi = 0.0, 9.5
    y_lo, y_hi = 0.0, 8.5
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # --- two nearly-parallel separating lines A and B ----------------------
    # Both have a moderate negative slope; they differ only in intercept.
    # Line B (upper): passes through the gap closer to the blue cluster.
    # Line A (lower): passes through the gap closer to the red cluster.
    # Slope chosen so both lines cleanly separate all points.
    xs = np.linspace(x_lo, x_hi, 300)

    slope = -0.62          # shared approximate slope (mirroring original)

    # Line B sits higher (more upper-left of the gap)
    intercept_b = 6.8
    yb = slope * xs + intercept_b

    # Line A sits lower (more lower-right of the gap)
    intercept_a = 5.4
    ya = slope * xs + intercept_a

    def clip_line(x, yv):
        """Keep only the portion of the line inside the axes view box."""
        mask = (yv >= y_lo) & (yv <= y_hi)
        return x[mask], yv[mask]

    xb_c, yb_c = clip_line(xs, yb)
    xa_c, ya_c = clip_line(xs, ya)

    ax.plot(xb_c, yb_c, color=LINE_COLOR, linewidth=2.4,
            solid_capstyle="round", zorder=2)
    ax.plot(xa_c, ya_c, color=LINE_COLOR, linewidth=2.4,
            solid_capstyle="round", zorder=2)

    # --- text labels A and B near the right end of each line ---------------
    # Place labels just to the right of where lines exit the plot on the
    # lower-right side (matching original layout).
    ax.text(8.4, slope * 8.4 + intercept_a + 0.3, "A",
            color=TEXT_COLOR, fontsize=18, fontweight="bold",
            ha="center", va="bottom", zorder=4)
    ax.text(6.5, slope * 6.5 + intercept_b + 0.3, "B",
            color=TEXT_COLOR, fontsize=18, fontweight="bold",
            ha="center", va="bottom", zorder=4)

    # --- axis styling: arrows, no numeric ticks ----------------------------
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    arrow_kw = dict(arrowstyle="-|>", color="#333333",
                    linewidth=1.5, mutation_scale=16)
    # x-axis arrow along the bottom
    ax.annotate("", xy=(x_hi, y_lo), xytext=(x_lo, y_lo),
                arrowprops=arrow_kw, zorder=1)
    # y-axis arrow along the left
    ax.annotate("", xy=(x_lo, y_hi), xytext=(x_lo, y_lo),
                arrowprops=arrow_kw, zorder=1)

    ax.set_title("线性可分时决策边界并不唯一", pad=10)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="png", dpi=200, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
