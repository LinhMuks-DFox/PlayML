"""Redraw of GradientDescent_10: the central-difference (symmetric quotient).

The original is a hand-drawn schematic illustrating numerical differentiation
via the *central difference* (symmetric difference quotient). On an
upward-opening loss parabola J, two points symmetric about a center x are
taken, x - h and x + h, and the secant through (x-h, f(x-h)) and
(x+h, f(x+h)) approximates the tangent slope (derivative) at x. The slope of
that secant is exactly (f(x+h) - f(x-h)) / (2h).

Concept faithfully reproduced with code-generated curves only (no dataset):

* The loss curve is the notes' parabola J(theta) = 2*theta**2 - 4*theta + 2,
  the same function used in 04-GradientDescent.md / the simulation notebook.
* The secant is built by a point-slope line whose slope is the central
  difference (f(x+h) - f(x-h)) / (2h), reusing the ``def line(point, X, k)``
  point-slope pattern from 04-GradientDescent.md (lines 446-448) and the
  tangent_line drawing范式 from the gradient-descent simulation notebook
  (cell 6/7). The line is drawn from the two end points, NOT from the
  analytic derivative, to expose the geometric meaning of the central
  difference.
* Three symmetric sample points (x-h, x, x+h) sit on the rising right branch
  of the parabola so the secant slope is positive, matching the original.
* Dashed projection lines drop from each point to the x-axis (labels
  x-h < x < x+h) and to the J-axis (labels f(x-h) < f(x) < f(x+h)).

Rendered as a clean schematic in the unified _style.py: top/right spines off,
arrowed x/J axes via annotate, no tick numbers (schematic style). The
hand-drawn jitter of the original is intentionally dropped — this is an
original matplotlib expression of the same concept.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/GradientDescent_10.png"

CURVE_COLOR = "#1A1A1A"          # near-black main loss curve (like original)
SECANT_COLOR = _style.PALETTE[2]  # green   #10B981  (matches original secant)
CENTER_COLOR = _style.PALETTE[0]  # blue    #2563EB  (matches original center line)
POINT_COLOR = "#1A1A1A"           # black sample points on the curve
PROJ_COLOR = "#6B7280"            # muted gray for projection dashes


def J(theta):
    """Notes' loss function: an upward-opening parabola, min at theta = 1."""
    return 2 * theta ** 2 - 4 * theta + 2


def line(point, X, k):
    """Point-slope line through ``point`` with slope ``k`` (04-GradientDescent.md).

    y - point_y = k * (x - point_x)  =>  y = k*(x - point_x) + point_y.
    """
    return k * (X - point[0]) + point[1]


def main():
    # --- Loss curve -------------------------------------------------------
    # Draw the full parabola so its vertex (theta = 1) sits LEFT of the
    # center x, and the three sample points land on the rising right branch.
    thetas = np.linspace(-0.55, 3.05, 400)
    losses = J(thetas)

    # Center point and a visible (schematic) finite step h.
    x = 2.0
    h = 0.5

    xm, x0, xp = x - h, x, x + h        # x-h, x, x+h
    fm, f0, fp = J(xm), J(x0), J(xp)     # f(x-h) < f(x) < f(x+h)

    # Central-difference slope = secant slope through the two outer points.
    k_central = (fp - fm) / (2 * h)

    fig, ax = _style.new_ax(figsize=(6.6, 5.8))
    ax.grid(False)  # schematic: no grid

    # --- Plot extents (origin at lower-left, generous margins for labels) -
    x_lo, x_hi = -0.95, 3.25
    y_lo, y_hi = -0.9, J(thetas).max() * 1.02

    # --- Main loss curve --------------------------------------------------
    ax.plot(thetas, losses, color=CURVE_COLOR, linewidth=2.6, zorder=3,
            solid_capstyle="round")

    # --- Vertical center reference line at x (blue, like the original) ----
    ax.plot([x0, x0], [y_lo, y_hi], color=CENTER_COLOR, linewidth=2.0,
            alpha=0.9, zorder=2)

    # --- Secant through (x-h, f(x-h)) and (x+h, f(x+h)) -------------------
    # Extend well beyond the two end points so it visibly "crosses" the frame.
    sec_x = np.linspace(x_lo + 0.05, x_hi - 0.05, 200)
    sec_y = line((xm, fm), sec_x, k_central)
    # Clip to the plotting window so the long secant does not blow up limits.
    inside = (sec_y >= y_lo) & (sec_y <= y_hi)
    ax.plot(sec_x[inside], sec_y[inside], color=SECANT_COLOR, linewidth=2.4,
            zorder=4, solid_capstyle="round")

    # --- Three sample points on the curve ---------------------------------
    ax.scatter([xm, x0, xp], [fm, f0, fp], s=46, color=POINT_COLOR,
               edgecolors="white", linewidths=1.0, zorder=6)

    # --- Dashed projections to the x-axis (bottom labels x-h, x, x+h) -----
    for px, py, lab in [(xm, fm, r"$x-h$"), (x0, f0, r"$x$"), (xp, fp, r"$x+h$")]:
        ax.plot([px, px], [y_lo, py], color=PROJ_COLOR, linewidth=1.1,
                linestyle="--", alpha=0.85, zorder=1)
        ax.annotate(lab, xy=(px, y_lo), xytext=(px, y_lo - 0.02),
                    ha="center", va="top", fontsize=11.5, color="#222222")

    # --- Dashed projections to the J-axis (left labels f(x-h..x+h)) -------
    for px, py, lab in [(xm, fm, r"$f(x-h)$"), (x0, f0, r"$f(x)$"),
                        (xp, fp, r"$f(x+h)$")]:
        ax.plot([x_lo, px], [py, py], color=PROJ_COLOR, linewidth=1.1,
                linestyle="--", alpha=0.85, zorder=1)
        ax.annotate(lab, xy=(x_lo, py), xytext=(x_lo - 0.04, py),
                    ha="right", va="center", fontsize=11.5, color="#222222")

    # --- Arrowed axes (origin at lower-left) ------------------------------
    # x-axis arrow.
    ax.annotate("", xy=(x_hi, y_lo), xytext=(x_lo, y_lo),
                arrowprops=dict(arrowstyle="-|>", color="#222222",
                                linewidth=1.8, mutation_scale=18), zorder=5)
    # J-axis arrow.
    ax.annotate("", xy=(x_lo, y_hi), xytext=(x_lo, y_lo),
                arrowprops=dict(arrowstyle="-|>", color="#222222",
                                linewidth=1.8, mutation_scale=18), zorder=5)
    ax.text(x_hi, y_lo - 0.02, r"$x$", ha="right", va="top", fontsize=13,
            color="#222222")
    ax.text(x_lo - 0.04, y_hi, r"$J$", ha="right", va="top", fontsize=13,
            color="#222222")

    # --- Formula annotation: secant slope = central difference ------------
    # Point near the upper end of the secant inside the frame.
    sx = sec_x[inside][-1]
    sy = sec_y[inside][-1]
    ax.annotate(
        r"割线斜率 $=\dfrac{f(x+h)-f(x-h)}{2h}\approx J'(x)$",
        xy=(sx - 0.18, sy - 0.18 * k_central),
        xytext=(0.30, y_hi * 0.80),
        ha="left", va="center", fontsize=11.5, color=SECANT_COLOR,
        arrowprops=dict(arrowstyle="-|>", color=SECANT_COLOR, linewidth=1.5,
                        mutation_scale=14, shrinkA=4, shrinkB=6,
                        connectionstyle="arc3,rad=-0.2"),
        zorder=7,
    )

    # --- Final cosmetics --------------------------------------------------
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("中心差分（对称差商）近似导数", fontsize=14, pad=12)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
