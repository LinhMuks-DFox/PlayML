"""Redraw of PCAAndGradientAscent_01: the PCA chapter intro scatter.

A purely illustrative 2D scatter for introducing PCA: a small set of
approximately 5-6 positively correlated points (lower-left to upper-right).
Horizontal axis = x (first feature), vertical axis = y (second feature).
This is the *first* concept figure of the PCA chapter, intentionally minimal:
just the scattered data points, with no fitting line, no projection, no arrows.

Data: 6 hardcoded points with a rough positive-correlation trend,
x in ~[1, 6], y in ~[1.5, 3.5], matching the original figure's sparse feel.

Style follows the unified _style.py (CJK-capable fonts, clean modern look).
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "PCAAndGradientAscent_01.png"
)

# Default matplotlib-blue scatter, as in the original figure.
POINT_COLOR = _style.PALETTE[0]  # blue #2563EB


def main():
    # 6 hardcoded illustrative points with positive correlation trend
    x = np.array([1.2, 2.1, 3.3, 4.0, 4.8, 5.7])
    y = np.array([1.6, 2.0, 2.4, 2.7, 3.1, 3.4])

    fig, ax = _style.new_ax(figsize=(5, 4))

    ax.scatter(
        x,
        y,
        s=50,
        c=POINT_COLOR,
        marker="o",
        zorder=3,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # No grid for minimal style
    ax.grid(False)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
