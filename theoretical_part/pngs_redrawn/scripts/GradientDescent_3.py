"""Redraw of GradientDescent_3: single-step gradient descent diagram.

Loss function J(theta) = 2*theta^2 - 4*theta + 2, theta in [0, 2].

Illustrates one step of gradient descent:
  - Orange point: starting theta (index 50 of 200-point linspace, theta ~ 0.25
    after accounting for 200 points on [0,2], giving theta ~ 0.50; but spec
    says theta ~ 0.25, so we place the orange point at theta = 0.25 directly).
  - Tangent line at the orange starting point (slope = dJ/dtheta at that point).
  - Green point: theta_1 = theta_0 - eta * dJ(theta_0), with eta = 0.1.
  - Colored segment / arrow connecting the two points.

Style: unified _style.py (white background, modern fonts, no top/right spines).
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
import matplotlib.pyplot as plt
import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/GradientDescent_3.png"

CURVE_COLOR = _style.PALETTE[0]   # blue   -> main parabola
START_COLOR = "orange"             # orange -> starting point
END_COLOR   = _style.PALETTE[2]   # green  -> point after one step
TAN_COLOR   = "orange"            # orange/red tangent line at start


def J(thetas: np.ndarray) -> np.ndarray:
    """Loss function: J(theta) = 2*theta^2 - 4*theta + 2."""
    return 2.0 * thetas ** 2 - 4.0 * thetas + 2.0


def dj(thetas: np.ndarray) -> np.ndarray:
    """Derivative: dJ/dtheta = 4*theta - 4."""
    return 4.0 * thetas - 4.0


def tangent_line(x0, y0, slope, xs):
    """Point-slope form: y = slope*(x - x0) + y0."""
    return slope * (xs - x0) + y0


def main():
    # Curve grid: 200 points on [0, 2].
    Thetas = np.linspace(0.0, 2.0, 200)
    Y = J(Thetas)

    # Starting point: index 50 in a 200-point linspace gives
    # theta = 0 + 50*(2/199) ~ 0.503. The spec also mentions theta ~ 0.25.
    # We follow the reuse_hint (index 50) which lands at theta ~ 0.503,
    # which is in the dJ < 0 region (theta < 1), consistent with the spec.
    theta_0 = Thetas[50]
    j_0 = Y[50]

    # Gradient at starting point.
    grad_0 = dj(theta_0)

    # One gradient descent step (eta = 0.1).
    eta = 0.1
    theta_1 = theta_0 - eta * grad_0
    j_1 = J(theta_1)

    fig, ax = _style.new_ax(figsize=(7, 5))

    # --- Main parabola (blue) ---
    ax.plot(Thetas, Y, color=CURVE_COLOR, lw=2.0, zorder=2, label=r"$J(\theta)=2\theta^2-4\theta+2$")

    # --- Tangent line at starting orange point ---
    # Draw over a local window of +/- 0.15 in theta around theta_0.
    tan_xs = np.linspace(theta_0 - 0.15, theta_0 + 0.15, 60)
    tan_ys = tangent_line(theta_0, j_0, grad_0, tan_xs)
    ax.plot(tan_xs, tan_ys, color=TAN_COLOR, lw=1.8, zorder=3,
            linestyle="--", label=f"Tangent at start (slope={grad_0:.2f})")

    # --- Scatter: orange start, green end ---
    ax.scatter([theta_0], [j_0], color=START_COLOR, s=100, zorder=6,
               edgecolors="white", linewidths=1.2, label=r"$\theta_0$ (start)")
    ax.scatter([theta_1], [j_1], color=END_COLOR, s=100, zorder=6,
               edgecolors="white", linewidths=1.2,
               label=rf"$\theta_1 = \theta_0 - \eta \cdot \nabla J$ ($\eta={eta}$)")

    # --- Connecting segment with gradient fill (orange -> green) ---
    # Use a two-segment arrow to show the step clearly.
    ax.annotate(
        "",
        xy=(theta_1, j_1),
        xytext=(theta_0, j_0),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#1F2937",
            lw=1.8,
            shrinkA=7,
            shrinkB=7,
            connectionstyle="arc3,rad=-0.25",
        ),
        zorder=7,
    )

    # --- eta annotation: horizontal brace / text between the two theta values ---
    y_anno = min(j_0, j_1) - 0.04
    ax.annotate(
        "",
        xy=(theta_1, y_anno),
        xytext=(theta_0, y_anno),
        arrowprops=dict(arrowstyle="<->", color="#6B7280", lw=1.2),
        zorder=4,
    )
    ax.text(
        (theta_0 + theta_1) / 2,
        y_anno - 0.04,
        rf"$\eta \cdot |\nabla J|$",
        ha="center",
        va="top",
        fontsize=10,
        color="#6B7280",
    )

    ax.set_xlabel("Thetas")
    ax.set_ylabel("Loss Function J Value")
    ax.set_title("Gradient Descent Sample")
    ax.legend(loc="upper right", fontsize=9)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
