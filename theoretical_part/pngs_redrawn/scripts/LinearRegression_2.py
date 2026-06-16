"""Redraw of LinearRegression_2: the linear-regression intro scatter plot.

A simple 2D scatter that opens the linear-regression chapter: discrete
"house area (x) vs house price (y)" samples that trend upward (positive
correlation) but do not sit exactly on a line, motivating the idea that we
need to fit a straight line through them.

This is the FIRST figure in the set, so it shows ONLY the raw sample points
(no red fit line yet -- that belongs to LinearRegression_3). The exact data
comes from the chp3 notebook toy dataset: x=[1,2,3,4,5], y=[1,3,2,3,5].

Style matches the unified _style.py (CJK-capable fonts, clean modern look).
The original had no title and no legend; we keep it minimal but add Chinese
axis labels for readability and consistency with the rest of the set.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/LinearRegression_2.png"

# Reuse the unified palette: blue dots, matching the original matplotlib look.
POINT_COLOR = _style.PALETTE[0]  # blue #2563EB


def build_dataset():
    """The exact chp3 toy dataset: upward trend, not perfectly collinear."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 3.0, 2.0, 3.0, 5.0])
    return x, y


def main():
    x, y = build_dataset()

    fig, ax = _style.new_ax(figsize=(6, 4.5))

    # Filled circles with a white edge for a clean, modern pop.
    ax.scatter(
        x,
        y,
        s=110,
        c=POINT_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
    )

    # Fixed plot range so the points sit centered, as in the original.
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xticks(range(0, 7))
    ax.set_yticks(range(0, 7))
    ax.set_aspect("equal", adjustable="box")

    # Chinese axis labels (CJK font configured by _style).
    ax.set_xlabel("房屋面积")
    ax.set_ylabel("房屋价格")

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
