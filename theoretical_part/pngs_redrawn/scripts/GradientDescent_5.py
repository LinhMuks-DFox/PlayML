"""Redraw of GradientDescent_5: "learning rate too large -> divergence".

Concept diagram for gradient descent when the learning rate eta is too big.
An upward-opening parabola J(theta) = (theta-2.5)^2 - 1 is the loss surface.
With eta=1.1, the iterates oscillate with growing amplitude, bouncing left and
right without converging.

Three blue filled dots (theta_0, theta_1, theta_2) are taken from the first
three positions in the iteration history (indices 0..2). Two red curved arc
arrows connect point0->point1 and point1->point2, visualizing the back-and-forth
oscillation. A Chinese annotation "eta 太大，甚至导致不收敛" appears at the top.

Data: fully synthetic.
  Curve: J(theta) = (theta-2.5)^2 - 1, theta in [-1, 6].
  Iterates: theta0=0, eta=1.1, 2 steps (producing 3 points).
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
import matplotlib.patches as mpatches

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/GradientDescent_5.png"

CURVE_COLOR = "#333333"           # dark gray parabola
POINT_COLOR = _style.PALETTE[0]   # blue filled iterate dots  #2563EB
ARROW_COLOR = _style.PALETTE[1]   # red curved arrows         #EF4444


def J(theta):
    return (theta - 2.5) ** 2 - 1.0


def dJ(theta):
    return 2.0 * (theta - 2.5)


def gradient_descent(theta0, eta, n_iters):
    theta = theta0
    history = [theta]
    for _ in range(n_iters):
        theta = theta - eta * dJ(theta)
        history.append(theta)
    return np.array(history)


def main():
    # Parabola curve — extend left to -2.0 so theta=-1.1 lands on the curve
    thetas = np.linspace(-2.0, 7.0, 400)
    ys = J(thetas)

    # Spec: eta=1.1, theta0=0, take first 3 points (indices 0, 1, 2)
    eta = 1.1
    theta_hist = gradient_descent(theta0=0.0, eta=eta, n_iters=2)
    # theta_hist has 3 elements: [0.0, theta_1, theta_2]
    j_hist = J(theta_hist)

    fig, ax = _style.new_ax(figsize=(8, 5))

    # Parabola (loss surface)
    ax.plot(thetas, ys, color=CURVE_COLOR, linewidth=2.2, zorder=1)

    # Three blue iterate dots
    ax.scatter(
        theta_hist,
        j_hist,
        s=140,
        c=POINT_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
    )

    # Red curved arc arrows: point0->point1, point1->point2
    # Alternate the arc direction so they visually arc over the parabola
    arc_rads = [0.35, -0.35]
    for k in range(len(theta_hist) - 1):
        rad = arc_rads[k % len(arc_rads)]
        ax.annotate(
            "",
            xy=(theta_hist[k + 1], j_hist[k + 1]),
            xytext=(theta_hist[k], j_hist[k]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=ARROW_COLOR,
                linewidth=2.0,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=10,
                shrinkB=10,
            ),
            zorder=4,
        )

    # Chinese title annotation at the top of the figure
    y_top = max(j_hist.max(), ys.max())
    ax.text(
        2.5,
        y_top + 1.5,
        r"$\eta$ 太大，甚至导致不收敛",
        ha="center",
        va="bottom",
        fontsize=15,
        color="#1F2937",
        zorder=6,
    )

    # Axis labels (spec requires theta and J(theta) labels)
    ax.set_xlabel(r"参数 $\theta$", fontsize=12)
    ax.set_ylabel(r"损失函数 $J(\theta)$", fontsize=12)

    # Schematic style: no ticks, no grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # Give headroom above the highest point + title text
    # x range must cover theta=-1.1 (leftmost iterate) with some margin
    ax.set_xlim(-2.5, 7.5)
    ax.set_ylim(ys.min() - 1.5, y_top + 3.5)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)
    print("eta=%.2f  theta_hist=%s" % (eta, np.round(theta_hist, 3)))
    print("j_hist=%s" % np.round(j_hist, 3))


if __name__ == "__main__":
    main()
