"""Redraw of Logisitic-Regression_1: the standard Sigmoid (logistic) curve.

The Sigmoid function sigma(t) = 1 / (1 + e^{-t}) squashes the arbitrary
real-valued linear output t = theta^T . x_b of a linear model into the
probability range [0, 1]. The original figure is the bare matplotlib default
plot of sigma over t in [-10, 10]: a single S-shaped curve, left end -> 0,
right end -> 1, value 0.5 at t = 0.

This redraw keeps the same concept but enhances readability with the unified
_style.py (clean modern look, CJK-capable fonts): axis labels, the two
asymptote lines y = 0 and y = 1, a horizontal decision-threshold line at
y = 0.5, a vertical line at t = 0, the highlighted key point (0, 0.5), and a
legend showing the closed-form formula. The sigmoid function below is the same
one used in the course code (playML/LogisticRegression._sigmoid).
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Logisitic-Regression_1.png"

CURVE_COLOR = _style.PALETTE[0]    # blue   #2563EB  -> sigmoid curve
THRESH_COLOR = _style.PALETTE[1]   # red    #EF4444  -> decision threshold 0.5
ASYMP_COLOR = _style.PALETTE[7]    # slate  #64748B  -> asymptotes / t = 0
POINT_COLOR = _style.PALETTE[4]    # violet #8B5CF6  -> key point (0, 0.5)


def _sigmoid(t):
    """Logistic sigmoid, identical to playML/LogisticRegression._sigmoid."""
    return 1. / (1. + np.exp(-t))


def main():
    # Dense sampling (500 points) over [-10, 10] for a smooth S-curve.
    t = np.linspace(-10, 10, 500)
    y = _sigmoid(t)

    fig, ax = _style.new_ax(figsize=(6.4, 4.4))

    # Asymptotes y = 0 and y = 1 (faint dotted slate lines).
    for y_asymp in (0.0, 1.0):
        ax.axhline(
            y_asymp,
            color=ASYMP_COLOR,
            linestyle=":",
            linewidth=1.3,
            alpha=0.7,
            zorder=1,
        )
    ax.text(
        -9.6, 1.0, r"$y \to 1$", color=ASYMP_COLOR, fontsize=10,
        ha="left", va="bottom", zorder=3,
    )
    ax.text(
        9.6, 0.0, r"$y \to 0$", color=ASYMP_COLOR, fontsize=10,
        ha="right", va="bottom", zorder=3,
    )

    # Decision threshold y = 0.5 (red dashed) and the vertical line t = 0.
    ax.axhline(
        0.5, color=THRESH_COLOR, linestyle="--", linewidth=1.4,
        alpha=0.85, zorder=2, label="决策阈值 0.5",
    )
    ax.axvline(
        0.0, color=ASYMP_COLOR, linestyle="--", linewidth=1.2,
        alpha=0.6, zorder=1,
    )

    # The sigmoid curve itself.
    ax.plot(
        t, y,
        color=CURVE_COLOR,
        linewidth=2.6,
        zorder=4,
        label=r"$\sigma(t) = \dfrac{1}{1 + e^{-t}}$",
    )

    # Key point (0, 0.5): t = 0 maps to probability 0.5.
    ax.scatter(
        [0.0], [0.5],
        s=95,
        color=POINT_COLOR,
        edgecolors="white",
        linewidths=1.6,
        zorder=5,
    )
    ax.annotate(
        r"$(0,\ 0.5)$",
        xy=(0.0, 0.5),
        xytext=(2.0, 0.30),
        fontsize=11,
        color=POINT_COLOR,
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="-|>",
            color=POINT_COLOR,
            linewidth=1.6,
            mutation_scale=15,
            shrinkA=2,
            shrinkB=6,
        ),
        zorder=6,
    )

    # Axis ranges: t in [-10, 10], probability in [0, 1] (small pad on y).
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])

    # Chinese labels (CJK font configured in _style).
    ax.set_xlabel(r"线性输出 $t = \theta^T \cdot x_b$")
    ax.set_ylabel(r"概率 $\hat{p} = \sigma(t)$")
    ax.set_title("Sigmoid 函数：将实数压缩到 [0, 1]")

    ax.legend(loc="upper left", framealpha=0.9)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
