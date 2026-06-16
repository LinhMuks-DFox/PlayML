"""Redraw of PCAAndGradientAscent_02: PCA dimension reduction by keeping
only the x feature (dropping y).

Concept figure for the PCA chapter. The blue dots are the original 2D
samples. The red dots are those same samples projected onto the x axis
(y set to 0), giving an intuitive picture of "reducing dimensionality =
squashing points onto an axis".

Style follows the unified _style.py (CJK-capable fonts, clean modern look).
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "PCAAndGradientAscent_02.png"
)

# Original colors: blue = original samples, red = projected samples.
ORIGINAL_COLOR = _style.PALETTE[0]   # blue #2563EB
PROJECTED_COLOR = _style.PALETTE[1]  # red  #EF4444

# ~6-8 representative 2D points with positive correlation trend.
# x range ~1..6, y range ~1.5..3.5, fixed seed for reproducibility.
np.random.seed(42)
n = 7
x_pts = np.linspace(1.5, 5.8, n) + np.random.randn(n) * 0.15
y_pts = 0.4 * x_pts + 1.0 + np.random.randn(n) * 0.18
# Clip to expected ranges
x_pts = np.clip(x_pts, 1.0, 6.2)
y_pts = np.clip(y_pts, 1.5, 3.5)

POINTS = np.column_stack([x_pts, y_pts])


def main():
    fig, ax = _style.new_ax(figsize=(6.4, 4.4))

    # Projected points: same x, y = 0
    proj_x = POINTS[:, 0]
    proj_y = np.zeros(len(POINTS))

    # Dashed gray guide lines from each original point down to the x axis,
    # emphasizing the "drop the y feature" action.
    for (px, py), qy in zip(POINTS, proj_y):
        ax.plot(
            [px, px],
            [py, qy],
            linestyle="--",
            color="#94A3B8",
            linewidth=1.2,
            alpha=0.8,
            zorder=1,
        )

    # Projected samples on the x axis (red) - drawn first so blue sits on top.
    ax.scatter(
        proj_x,
        proj_y,
        s=90,
        c=PROJECTED_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
        label="丢弃 y 后的数据点（投影到 x 轴）",
    )

    # Original 2D samples (blue).
    ax.scatter(
        POINTS[:, 0],
        POINTS[:, 1],
        s=90,
        c=ORIGINAL_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.2,
        zorder=4,
        label="原始数据点",
    )

    # Horizontal reference line at y = 0 (projection target axis).
    ax.axhline(y=0, color="#64748B", linewidth=1.0, alpha=0.6, zorder=0)

    # Axis ranges: x ~ 0.8..6.5, y slightly below 0 so red dots are visible.
    ax.set_xlim(0.8, 6.5)
    ax.set_ylim(-0.35, 3.9)

    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    ax.set_xlabel("特征 x")
    ax.set_ylabel("特征 y")
    ax.set_title("只保留 x 特征（丢弃 y）")

    ax.legend(loc="upper left", framealpha=0.9)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
