"""Redraw of GradientDescent_9: "numerical differentiation (numerical diff)".

Concept diagram explaining the definition of the derivative as the limit of a
*secant* slope:

        f'(x) ~= [ f(x + h) - f(x) ] / h .

An upward-opening parabola f is drawn (black). On its right (rising) arm we pick
two neighbouring points (x, f(x)) and (x + h, f(x + h)) with a *finite, visible*
step h, so the two points stay clearly separated -- this is the whole point of
"numerical" differentiation: we approximate the tangent by a secant through two
sampled points rather than by an analytic derivative.

Through those two points we draw the secant line (green), extended past both
ends of the curve, exactly as in the original hand-drawn sketch. Light dashed
guide lines drop from each point horizontally to the y axis (labelling f(x) and
f(x+h)) and vertically to the x axis (labelling x and x+h). A blue vertical
reference line marks the location of the minimum (vertex), matching the original.

Function + point-slope drawing reuse the teaching setup from
notebooks/chp4-Gradient-Descent-And-Linear-Regression/01-Gradient-Descent-Simulations.ipynb
(Cell 6 / Cell 7: J(theta) = (theta - 2.5)**2 - 1 parabola + tangent_line point
-slope helper). Here the analytic derivative dJ is replaced by the two-point
secant slope (f(x+h) - f(x)) / h, which is the numerical-differentiation idea.

Style follows the unified _style.py (CJK-capable fonts, clean modern look). This
is a schematic, so the theta / f axes carry no numeric ticks; instead the left
and bottom spines act as arrowed coordinate axes, echoing the hand-drawn look.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/GradientDescent_9.png"

CURVE_COLOR = "#1A1A1A"          # near-black parabola (main curve)
SECANT_COLOR = _style.PALETTE[2]  # green secant line (matches original)
POINT_COLOR = "#1A1A1A"          # black sampled points on the curve
REF_COLOR = _style.PALETTE[0]    # blue vertical reference line at the minimum
GUIDE_COLOR = "#94A3B8"          # light slate dashed guide lines


def f(x):
    """Upward-opening quadratic; vertex at x = 2.5, minimum f = -1."""
    return (x - 2.5) ** 2 - 1.0


def secant_slope(x, h):
    """Numerical-differentiation slope: (f(x+h) - f(x)) / h."""
    return (f(x + h) - f(x)) / h


def main():
    # --- Curve and the two sampled points ----------------------------------
    xs = np.linspace(-1.0, 6.0, 400)
    ys = f(xs)

    # Two neighbouring points on the RIGHT (rising) arm so that
    # f(x+h) > f(x) and the secant slope is positive (matches original).
    # h is a *finite, visible* value (=1.0) so the points are clearly apart.
    x0 = 3.4
    h = 1.0
    x1 = x0 + h
    y0, y1 = f(x0), f(x1)
    k = secant_slope(x0, h)  # secant slope used as the numerical derivative

    fig, ax = _style.new_ax(figsize=(6.8, 5.4))

    # --- Main parabola ------------------------------------------------------
    ax.plot(xs, ys, color=CURVE_COLOR, linewidth=2.4, zorder=3)

    # --- Secant line through the two points, extended past both ends --------
    # Point-slope form y = y0 + k * (x - x0), drawn across a wide x span so it
    # sticks out beyond the curve on both sides (as in the original sketch).
    sec_x = np.array([1.2, 6.0])
    sec_y = y0 + k * (sec_x - x0)
    ax.plot(sec_x, sec_y, color=SECANT_COLOR, linewidth=2.2, zorder=2)

    # --- Vertical reference line at the minimum (vertex x = 2.5) ------------
    ax.axvline(2.5, color=REF_COLOR, linewidth=2.0, zorder=1)

    # --- Guide lines (light dashed) -----------------------------------------
    # Horizontal: each point -> y axis (x = x_left). Vertical: each -> x axis.
    x_axis_y = ys.min() - 1.6   # where the bottom "x axis" sits
    y_axis_x = -1.0             # where the left "y axis" sits

    for px, py in ((x0, y0), (x1, y1)):
        # horizontal guide to the y axis
        ax.plot([y_axis_x, px], [py, py], color=GUIDE_COLOR,
                linestyle="--", linewidth=1.2, zorder=1)
        # vertical guide down to the x axis
        ax.plot([px, px], [x_axis_y, py], color=GUIDE_COLOR,
                linestyle="--", linewidth=1.2, zorder=1)

    # --- The two sampled points ---------------------------------------------
    ax.scatter([x0, x1], [y0, y1], s=80, c=POINT_COLOR,
               edgecolors="white", linewidths=1.0, zorder=5)

    # --- Axis labels on the spines: f(x), f(x+h), x, x+h --------------------
    # On the y axis (left): f(x+h) sits above f(x) since f(x+h) > f(x).
    ax.text(y_axis_x - 0.18, y0, r"$f(x)$", ha="right", va="center",
            fontsize=14, color=CURVE_COLOR)
    ax.text(y_axis_x - 0.18, y1, r"$f(x+h)$", ha="right", va="center",
            fontsize=14, color=CURVE_COLOR)
    # On the x axis (bottom): x then x+h.
    ax.text(x0, x_axis_y - 0.30, r"$x$", ha="center", va="top",
            fontsize=14, color=CURVE_COLOR)
    ax.text(x1, x_axis_y - 0.30, r"$x+h$", ha="center", va="top",
            fontsize=14, color=CURVE_COLOR)

    # --- Coordinate axes drawn as arrowed spines ----------------------------
    # Hide matplotlib's default frame and draw our own arrowed x / y axes,
    # echoing the hand-drawn arrows in the original sketch.
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    x_top = ys.max() + 1.2
    x_right = 6.4
    # y axis (vertical arrow, upward)
    ax.annotate("", xy=(y_axis_x, x_top), xytext=(y_axis_x, x_axis_y),
                arrowprops=dict(arrowstyle="-|>", color=CURVE_COLOR, lw=2.0),
                zorder=4)
    # x axis (horizontal arrow, rightward)
    ax.annotate("", xy=(x_right, x_axis_y), xytext=(y_axis_x, x_axis_y),
                arrowprops=dict(arrowstyle="-|>", color=CURVE_COLOR, lw=2.0),
                zorder=4)

    # --- Title (concept) -----------------------------------------------------
    ax.set_title("数值微分：用割线斜率近似导数", fontsize=15, pad=12)
    # Small caption recalling the definition formula.
    ax.text(
        2.0, x_top - 0.4,
        r"$f'(x)\approx \dfrac{f(x+h)-f(x)}{h}$",
        ha="center", va="top", fontsize=13, color=SECANT_COLOR,
    )

    # --- Limits with a little padding ---------------------------------------
    ax.set_xlim(y_axis_x - 1.0, x_right + 0.2)
    ax.set_ylim(x_axis_y - 0.4, x_top + 0.3)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="png")
    print("saved:", OUT_PATH)
    print("x0=%.2f h=%.2f -> secant slope k=%.3f" % (x0, h, k))


if __name__ == "__main__":
    main()
