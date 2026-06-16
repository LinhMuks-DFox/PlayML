"""Redraw of GradientDescent_1: the single-parameter loss curve.

A simple upward-opening parabola J(theta) = 2*theta**2 - 4*theta + 2 over
theta in [0, 2], used to introduce the idea that gradient descent = finding
the minimum of the loss curve. The curve has its unique minimum at
theta = 1.0, J = 0.

This is the introductory / "guide" figure for the gradient descent section.
It shows only the curve itself (no iteration points, no tangent lines, no
descent arrows). A small dot at the minimum optionally marks theta* = 1.

Faithful to the original markdown plotting code (04-GradientDescent.md,
J = 2*theta**2 - 4*theta + 2, 200 sample points). Styled with the unified
_style.py (clean modern look, CJK-capable fonts).
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/GradientDescent_1.png"

CURVE_COLOR = _style.PALETTE[0]  # blue  #2563EB
MIN_COLOR = _style.PALETTE[1]    # red   #EF4444


def J(thetas):
    """Original loss function from the notes: an upward-opening parabola."""
    return 2 * thetas ** 2 - 4 * thetas + 2


def main():
    # 200 points on [0, 2] (closed interval), matching the spec.
    thetas = np.linspace(0, 2, 200)
    losses = J(thetas)

    fig, ax = _style.new_ax(figsize=(6, 4.5))

    # The loss curve itself (single blue parabola, no extra decorations).
    ax.plot(
        thetas,
        losses,
        color=CURVE_COLOR,
        linewidth=2,
        zorder=2,
    )

    # Optional: mark the unique minimum at theta=1, J=0.
    theta_min = 1.0
    j_min = float(J(np.array(theta_min)))
    ax.scatter(
        [theta_min],
        [j_min],
        s=80,
        color=MIN_COLOR,
        edgecolors="white",
        linewidths=1.4,
        zorder=4,
        label=r"$\theta^* = 1$",
    )

    # Axis ranges matching the original (~[0, 2] x [0, 2]).
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)

    # Axis labels and title (English, matching the spec).
    ax.set_xlabel("Theta")
    ax.set_ylabel("Loss Function J Value")
    ax.set_title("Gradient Descent Sample")

    ax.legend(loc="upper center", framealpha=0.9)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
