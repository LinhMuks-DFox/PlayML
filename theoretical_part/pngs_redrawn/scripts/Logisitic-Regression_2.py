"""Redraw of Logisitic-Regression_2: the natural-log base curve.

This is a foundational helper figure in the logistic-regression notes used to
build up the loss-function derivation. It shows the natural logarithm
``y = log(x)`` over ``x in (0, 10]``:

  * the curve passes through ``(1, 0)``,
  * as ``x -> 0+`` it plunges toward negative infinity (steep left tail),
  * for ``x > 1`` it rises slowly and monotonically (``ln 10 ~= 2.30``).

It sets up the later explanation of the ``-log(p)`` loss term.

Style follows the unified ``_style.py``.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Logisitic-Regression_2.png"

CURVE_COLOR = _style.PALETTE[0]   # blue #2563EB, matching the original
POINT_COLOR = _style.PALETTE[1]   # red #EF4444, to highlight (1, 0)


def build_curve():
    """Natural-log curve on (0, 10].

    Start at x=0.01 so log(0.01) ~ -4.6, matching the original y-axis bottom.
    500 points keeps the curve smooth.
    """
    x = np.linspace(0.01, 10.0, 500)
    y = np.log(x)
    return x, y


def main():
    x, y = build_curve()

    fig, ax = _style.new_ax(figsize=(6, 4.5))

    ax.plot(x, y, color=CURVE_COLOR, zorder=2)

    # Emphasize the (1, 0) crossing point — referenced in notes as "经过 (1, 0)".
    ax.scatter(
        [1.0],
        [0.0],
        s=55,
        c=POINT_COLOR,
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
    )
    ax.annotate(
        "(1, 0)",
        xy=(1.0, 0.0),
        xytext=(1.6, -0.9),
        fontsize=10,
        color=POINT_COLOR,
        arrowprops=dict(arrowstyle="->", color=POINT_COLOR, lw=1.0),
    )

    # Fix axes ranges to match the original figure.
    ax.set_xlim(0, 10)
    ax.set_ylim(-4.6, 2.3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("log函数的曲线")

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
