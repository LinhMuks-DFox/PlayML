"""Redraw of LinearRegression_4: concept diagram for simple linear regression.

Shows scattered house-area / house-price samples and a red fit line y = ax + b.
For one selected sample point x^(i) the diagram uses dashed projection lines to
illustrate the gap between the algorithm's prediction ŷ^(i) (point on the line)
and the true value y^(i) (the actual data point).

Key annotation elements
------------------------
* Vertical dashed line: x^(i) down to x-axis.
* Horizontal dashed line: ŷ^(i) from the line intersection to the y-axis.
* Horizontal dashed line: y^(i) from the true point to the y-axis.
* Red arrow + label: "算法给出的预测"  → ŷ^(i) on y-axis.
* Red arrow + label: "实际房屋价格"    → y^(i) on y-axis.
* Red arrow + label: "样本房屋面积"    → x^(i) on x-axis.
* Formula text (LaTeX): y = ax + b and ŷ^(i) = ax^(i) + b.
* Red title "简单线性回归" at top.

Fixes vs previous version
--------------------------
1. Selected sample point is ABOVE the regression line (yi > yhat), matching the
   original lecture diagram and making ŷ^(i) clearly below y^(i) on the y-axis.
2. Title "简单线性回归" added in red at the top of the figure.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style
from _style import plt

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/LinearRegression_4.png"
)

# --- Colors -------------------------------------------------------------------
POINT_COLOR = _style.PALETTE[0]   # blue  #2563EB
LINE_COLOR  = _style.PALETTE[1]   # red   #EF4444
ANNOT_COLOR = _style.PALETTE[1]   # red, for arrow annotations
AXIS_COLOR  = "#222222"
DASH_COLOR  = "#888888"


def main():
    # --- Fit-line parameters (y = a·x + b) ------------------------------------
    a, b = 0.9, 0.5

    # --- Sample data (6 points scattered around the line) --------------------
    # Selected sample is at xi=3.8, placed ABOVE the regression line.
    # Line value at xi=3.8: yhat = 0.9*3.8 + 0.5 = 3.92
    # We set yi = 5.0, which is clearly above yhat. This means on the y-axis:
    #   y^(i) = 5.0  (higher)
    #   ŷ^(i) = 3.92 (lower)
    # The gap 5.0 - 3.92 ≈ 1.1 is clearly visible, correctly illustrating
    # that the true value exceeds the prediction.
    xs = np.array([1.5, 2.8, 3.8, 5.1, 6.4, 7.5])
    ys = np.array([1.1, 2.0, 5.0, 5.3, 6.1, 5.8])

    # Selected sample point (the one with the annotations)
    xi   = 3.8
    yi   = 5.0         # true value – above the line
    yhat = a * xi + b  # predicted value on the line = 3.92

    fig, ax = _style.new_ax(figsize=(8, 6.2))

    # Add red title as figure suptitle to match original lecture slide
    fig.suptitle("简单线性回归", fontsize=22, color=LINE_COLOR,
                 fontweight="bold", y=0.97)

    # --- Concept-diagram style: no grid, no spines, no tick numbers -----------
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot extents with margins for arrowed axes and callout labels.
    x_min, x_max = 0.0, 9.4
    y_min, y_max = 0.0, 9.2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # --- Arrowed axes (origin placed a little inside the data region) ---------
    ox, oy = 0.7, 0.5
    arrow_kw = dict(arrowstyle="-|>", color=AXIS_COLOR, linewidth=1.8,
                    mutation_scale=18)
    # x-axis arrow
    ax.annotate("", xy=(x_max - 0.15, oy), xytext=(ox, oy), arrowprops=arrow_kw)
    # y-axis arrow
    ax.annotate("", xy=(ox, y_max - 0.15), xytext=(ox, oy), arrowprops=arrow_kw)

    # Axis end labels: italic x, y
    ax.text(x_max, oy - 0.04, r"$x$", fontsize=18, ha="left",
            va="center", color=AXIS_COLOR)
    ax.text(ox - 0.06, y_max, r"$y$", fontsize=18, ha="right",
            va="bottom", color=AXIS_COLOR)

    # --- Red fit line y = a·x + b -------------------------------------------
    line_x = np.array([ox + 0.1, x_max - 0.3])
    line_y = a * line_x + b
    ax.plot(line_x, line_y, color=LINE_COLOR, linewidth=2.6, zorder=2)

    # --- Dashed projection lines for the selected sample ---------------------
    dash_style = (0, (4, 3))
    lw_dash = 1.3

    # Vertical: from x-axis up to the predicted point on the line at xi
    ax.plot([xi, xi], [oy, yhat], linestyle=dash_style,
            color=DASH_COLOR, linewidth=lw_dash, zorder=1)

    # Horizontal: from predicted point on the line → y-axis at ŷ^(i)
    ax.plot([ox, xi], [yhat, yhat], linestyle=dash_style,
            color=DASH_COLOR, linewidth=lw_dash, zorder=1)

    # Horizontal: from true sample point → y-axis at y^(i)
    ax.plot([ox, xi], [yi, yi], linestyle=dash_style,
            color=DASH_COLOR, linewidth=lw_dash, zorder=1)

    # --- Scatter the sample points -------------------------------------------
    ax.scatter(xs, ys, s=240, c=POINT_COLOR, marker="o",
               edgecolors="white", linewidths=1.6, zorder=4)

    # --- Tick labels for the three highlighted positions ----------------------
    ax.text(xi, oy - 0.18, r"$x^{(i)}$", fontsize=15, ha="center",
            va="top", color=AXIS_COLOR)
    ax.text(ox - 0.18, yhat, r"$\hat{y}^{(i)}$", fontsize=15, ha="right",
            va="center", color=AXIS_COLOR)
    ax.text(ox - 0.18, yi, r"$y^{(i)}$", fontsize=15, ha="right",
            va="center", color=AXIS_COLOR)

    # --- Formula annotations in the right-hand region ------------------------
    ax.text(5.5, 3.0, r"$y = ax + b$", fontsize=20, ha="left", va="center",
            color=AXIS_COLOR)
    ax.text(5.5, 1.8, r"$\hat{y}^{(i)} = ax^{(i)} + b$", fontsize=20,
            ha="left", va="center", color=AXIS_COLOR)

    # --- Red callout annotations with arrows ---------------------------------
    annot_kw = dict(fontsize=12, color=ANNOT_COLOR, ha="center", va="center")
    arrow_red = dict(arrowstyle="->", color=ANNOT_COLOR, linewidth=1.6)

    # "算法给出的预测" → ŷ^(i) on the y-axis
    # ŷ^(i) = 3.92, so text is placed above-left and arrow points to y-axis tick
    ax.annotate("算法给出的预测",
                xy=(ox + 0.05, yhat),
                xytext=(2.5, 6.5),
                arrowprops=arrow_red, **annot_kw)

    # "实际房屋价格" → y^(i) on the y-axis
    # y^(i) = 5.0, text is placed to the lower-left, arrow points to y-axis tick
    ax.annotate("实际房屋价格",
                xy=(ox + 0.05, yi),
                xytext=(2.2, 7.7),
                arrowprops=arrow_red, **annot_kw)

    # "样本房屋面积" → x^(i) on the x-axis
    ax.annotate("样本房屋面积",
                xy=(xi, oy + 0.05),
                xytext=(5.8, 1.0),
                arrowprops=arrow_red, **annot_kw)

    # Save using _style.finalize (handles tight_layout, dpi=200, closes fig).
    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
