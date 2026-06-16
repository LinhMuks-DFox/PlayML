"""Redraw of LinearRegression_3: the "best-fit line" of simple linear regression.

Five blue sample points (house area -> price) scattered on a 2D plane, with a
red straight line passing through them, illustrating how linear regression fits
the linear relationship between feature and label with a single line.

Data is the fixed teaching toy set from
notebooks/chp3-Linear-Regression/01-Simple-Linear-Regression-Implementation.ipynb:
    x = [1, 2, 3, 4, 5], y = [1, 3, 2, 3, 5]
The red line is the least-squares fit recomputed here (not hand-drawn):
    a = sum((xi - x_mean)(yi - y_mean)) / sum((xi - x_mean)^2)
    b = y_mean - a * x_mean
which yields a = 0.8, b = 0.4 for this data, matching the original screenshot.

Style follows the unified _style.py (CJK-capable fonts, clean modern look).
Plot bounds kept at [0, 6, 0, 6] to match the original proportions. Chinese
axis labels + legend are added to raise originality (the original was a bare
axes), without changing the concept.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/LinearRegression_3.png"

SAMPLE_COLOR = _style.PALETTE[0]  # blue #2563EB
LINE_COLOR = _style.PALETTE[1]    # red  #EF4444


def fit_simple_linear(x, y):
    """Least-squares slope/intercept for simple (1D) linear regression.

    a = sum((xi - x_mean)(yi - y_mean)) / sum((xi - x_mean)^2)
    b = y_mean - a * x_mean
    """
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    a = num / den
    b = y_mean - a * x_mean
    return a, b


def main():
    # Fixed teaching toy data (cell 2 of the reference notebook).
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 3.0, 2.0, 3.0, 5.0])

    a, b = fit_simple_linear(x, y)  # a ~= 0.8, b ~= 0.4
    y_hat = a * x + b

    fig, ax = _style.new_ax(figsize=(6.4, 4.8))

    # Blue sample scatter: filled circles with a white edge for a clean pop.
    ax.scatter(
        x,
        y,
        s=90,
        c=SAMPLE_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
        label="样本",
    )

    # Red least-squares fit line, spanning the data range.
    ax.plot(
        x,
        y_hat,
        color=LINE_COLOR,
        linewidth=2.2,
        zorder=2,
        label="拟合直线",
    )

    # Fixed bounds to match the original figure proportions and whitespace.
    ax.axis([0, 6, 0, 6])
    ax.set_aspect("equal", adjustable="box")

    # Keep the simple style: light grid from _style is fine; integer ticks.
    ax.set_xticks(range(0, 7))
    ax.set_yticks(range(0, 7))

    ax.set_xlabel("房屋面积")
    ax.set_ylabel("价格")

    ax.legend(loc="upper left", framealpha=0.9)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH, "| a=%.3f b=%.3f" % (a, b))


if __name__ == "__main__":
    main()
