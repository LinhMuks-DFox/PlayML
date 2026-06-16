"""Redraw of GradientDescent_2: tangent lines at three points on the parabola.

Loss function J(theta) = 2*theta^2 - 4*theta + 2, theta in [0, 2].
Three points are selected at indices 50 / 100 / 150 of a 201-point linspace,
corresponding to theta ~ 0.5 / 1.0 / 1.5, where dJ/dtheta < 0 / = 0 / > 0.
A local tangent segment is drawn at each point to show the sign of the slope.

Spec requirements:
  - English axis labels: "Thetas" / "Loss Function J Value"
  - English title: "Gradient Descent Sample"
  - No legend
  - Colors: blue curve, orange/green/red for the three points and tangents
  - Figure size 6x4 inches
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/GradientDescent_2.png"

# Semantic color choices from the unified palette.
CURVE_COLOR = _style.PALETTE[0]   # blue  -> main parabola
NEG_COLOR   = _style.PALETTE[3]   # amber/orange -> dJ < 0  (theta ~ 0.5)
ZERO_COLOR  = _style.PALETTE[2]   # green        -> dJ = 0  (theta = 1.0, min)
POS_COLOR   = _style.PALETTE[1]   # red          -> dJ > 0  (theta ~ 1.5)


def J(thetas):
    """J(theta) = 2*theta^2 - 4*theta + 2."""
    return 2.0 * thetas ** 2 - 4.0 * thetas + 2.0


def dj(thetas):
    """dJ/dtheta = 4*theta - 4."""
    return 4.0 * thetas - 4.0


def tangent_line(point_x, point_y, slope, xs):
    """Point-slope form: y = slope*(x - point_x) + point_y."""
    return slope * (xs - point_x) + point_y


def main():
    # 201-point linspace as specified; indices 50/100/150 give theta 0.5/1.0/1.5.
    Thetas = np.linspace(0.0, 2.0, 201)
    Y = J(Thetas)

    # Three representative points.
    dj_ls_0_point = (Thetas[50],  Y[50])   # theta ~ 0.5, dJ < 0
    dj_eq_0_point = (Thetas[100], Y[100])  # theta = 1.0, dJ = 0
    dj_gt_0_point = (Thetas[150], Y[150])  # theta ~ 1.5, dJ > 0

    slope_neg  = dj(dj_ls_0_point[0])
    slope_zero = dj(dj_eq_0_point[0])
    slope_pos  = dj(dj_gt_0_point[0])

    fig, ax = _style.new_ax(figsize=(6, 4))

    # Main parabola (blue).
    ax.plot(Thetas, Y, color=CURVE_COLOR, linewidth=2.0, zorder=2)

    # Tangent segments drawn on local windows (indices +-25, i.e. +-0.25 in theta).
    ax.plot(
        Thetas[25:75],
        tangent_line(dj_ls_0_point[0], dj_ls_0_point[1], slope_neg, Thetas[25:75]),
        color=NEG_COLOR, linewidth=2.0, zorder=3,
    )
    ax.plot(
        Thetas[75:125],
        tangent_line(dj_eq_0_point[0], dj_eq_0_point[1], slope_zero, Thetas[75:125]),
        color=ZERO_COLOR, linewidth=2.0, zorder=3,
    )
    ax.plot(
        Thetas[125:175],
        tangent_line(dj_gt_0_point[0], dj_gt_0_point[1], slope_pos, Thetas[125:175]),
        color=POS_COLOR, linewidth=2.0, zorder=3,
    )

    # Scatter points on the curve.
    ax.scatter(*dj_ls_0_point, s=90, color=NEG_COLOR,
               edgecolors="white", linewidths=1.2, zorder=4)
    ax.scatter(*dj_eq_0_point, s=90, color=ZERO_COLOR,
               edgecolors="white", linewidths=1.2, zorder=4)
    ax.scatter(*dj_gt_0_point, s=90, color=POS_COLOR,
               edgecolors="white", linewidths=1.2, zorder=4)

    # Axis labels and title (all English, no legend).
    ax.set_xlabel("Thetas")
    ax.set_ylabel("Loss Function J Value")
    ax.set_title("Gradient Descent Sample")

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
