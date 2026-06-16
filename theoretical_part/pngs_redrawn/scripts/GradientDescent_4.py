"""Redraw of GradientDescent_4: "eta too small -> slow convergence".

Concept diagram showing gradient descent with a very small learning rate
on the convex quadratic loss J(theta) = (theta - 2.5)^2 - 1.

QA fixes applied (v4):
- Use ax.annotate with arrowstyle='fancy' which draws a proper filled-arrow
  patch (not just a triangular marker), and a large rad so the arc body is
  clearly visible.  The annotation coordinate system is 'data', so positions
  are exact.
- eta=0.05, 6 steps (7 points): individually legible steps on the left arm.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/GradientDescent_4.png"
)

CURVE_COLOR = "#2D2D2D"
POINT_COLOR = _style.PALETTE[0]   # blue  #2563EB
ARROW_COLOR = _style.PALETTE[1]   # red   #EF4444


def J(theta):
    return (theta - 2.5) ** 2 - 1.0


def dJ(theta):
    return 2.0 * (theta - 2.5)


def gradient_descent(theta0, eta, n_steps):
    theta = float(theta0)
    history = [theta]
    for _ in range(n_steps):
        theta = theta - eta * dJ(theta)
        history.append(theta)
    return np.array(history)


def main():
    plot_x = np.linspace(-1.0, 6.0, 400)
    plot_y = J(plot_x)

    eta = 0.05
    theta0 = 0.0
    n_steps = 6
    theta_hist = gradient_descent(theta0, eta, n_steps)
    j_hist = J(theta_hist)

    fig, ax = _style.new_ax(figsize=(6.8, 4.8))

    ax.plot(plot_x, plot_y, color=CURVE_COLOR, linewidth=2.2, zorder=2)

    ax.scatter(
        theta_hist, j_hist,
        s=90, c=POINT_COLOR,
        marker="o", edgecolors="white", linewidths=1.3,
        zorder=4,
    )

    # Arrows: use 'fancy' arrowstyle which renders a smooth filled arrowhead
    # (not a triangular marker).  connectionstyle arc3 with rad=-0.5 gives a
    # clear outward bow to the left of the travel direction.
    for i in range(len(theta_hist) - 1):
        ax.annotate(
            "",
            xy=(theta_hist[i + 1], j_hist[i + 1]),
            xytext=(theta_hist[i], j_hist[i]),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(
                arrowstyle="fancy",
                color=ARROW_COLOR,
                fc=ARROW_COLOR,
                ec=ARROW_COLOR,
                mutation_scale=16,
                connectionstyle="arc3,rad=-0.5",
                shrinkA=6,
                shrinkB=6,
            ),
            zorder=5,
        )

    ax.set_xlim(-1.3, 5.5)
    ax.set_ylim(J(2.5) - 2.0, J(-1.0) + 5.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("参数 θ", fontsize=12)
    ax.set_ylabel("损失函数 J", fontsize=12)

    top_y = J(-1.0)
    ax.text(
        1.4, top_y + 4.2,
        r"$\eta$ 太小，减慢收敛学习速度",
        ha="center", va="center",
        fontsize=14, color="#1F2937", fontweight="bold",
    )

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)
    print(f"eta={eta}  theta0={theta0}  steps={n_steps}")
    print("theta_hist:", np.round(theta_hist, 4))
    print("j_hist:    ", np.round(j_hist, 4))


if __name__ == "__main__":
    main()
