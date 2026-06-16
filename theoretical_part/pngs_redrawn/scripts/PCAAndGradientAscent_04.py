"""Redraw of PCAAndGradientAscent_04: the PCA motivation figure.

Concept figure for the PCA chapter. A handful of positively correlated 2D
samples (blue dots) are shown with a blue diagonal line threading through the
point cloud from lower-left to upper-right. This visualizes PCA's core
motivation: rather than projecting onto the x axis (_02) or the y axis (_03),
we seek a slanted axis along which the samples spread out the most -- the
first principal component direction. Projecting onto this axis preserves
inter-sample distances and distinguishability far better than either
coordinate axis alone.

Style follows the unified _style.py (CJK-capable fonts, clean modern look).
No grid, no legend -- matching the minimal style of the _01 sibling figure.
Data is the same explicit 5-point POINTS array used in 02/03 so the four
figures (_01 through _04) tell one continuous story over the same point cloud.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "PCAAndGradientAscent_04.png"
)

# Blue for both points and the PCA axis line, per spec (蓝色实线).
POINT_COLOR = _style.PALETTE[0]   # #2563EB  -- scatter dots
LINE_COLOR  = _style.PALETTE[0]   # #2563EB  -- first principal component axis

# Shared 2D points: same as in figures 02 and 03 for narrative consistency.
POINTS = np.array(
    [
        [1.5, 1.5],
        [2.0, 1.8],
        [4.0, 3.0],
        [5.0, 3.5],
        [6.0, 3.0],
    ]
)


def main():
    # Fit a line through the cloud with np.polyfit (illustrative PCA axis).
    slope, intercept = np.polyfit(POINTS[:, 0], POINTS[:, 1], deg=1)

    fig, ax = _style.new_ax(figsize=(5, 4))

    # Draw the diagonal PCA axis line spanning well beyond the point cloud.
    x_lo, x_hi = 0.8, 6.8
    xs_line = np.array([x_lo, x_hi])
    ys_line = slope * xs_line + intercept
    ax.plot(
        xs_line,
        ys_line,
        color=LINE_COLOR,
        linewidth=2.0,
        zorder=2,
    )

    # Scatter the 5 illustrative data points.
    ax.scatter(
        POINTS[:, 0],
        POINTS[:, 1],
        s=70,
        c=POINT_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
    )

    # Axis limits matching the series (_01 through _03 share this range).
    ax.set_xlim(0.5, 7.0)
    ax.set_ylim(0.8, 4.0)

    # Minimal axis labels (no grid, no legend, per spec).
    ax.set_xlabel("特征 1")
    ax.set_ylabel("特征 2")
    ax.grid(False)

    # Optional lightweight annotation pointing at the PCA axis direction.
    x_anno = 5.5
    y_anno = slope * x_anno + intercept
    ax.annotate(
        "第一主成分方向",
        xy=(x_anno, y_anno),
        xytext=(3.2, 3.65),
        fontsize=9,
        color="#444444",
        arrowprops=dict(
            arrowstyle="->",
            color="#888888",
            linewidth=1.0,
        ),
        zorder=5,
    )

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
