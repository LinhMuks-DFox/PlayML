"""Redraw of Polynomial-Regression-and-Model-Generalization_9.

Concept diagram for *ridge (L2) regularized* gradient descent in parameter
space (theta_1, theta_2).

The L2 penalty gradient is alpha * theta, which is non-zero in *every*
coordinate as long as theta != 0. Pure gradient descent therefore shrinks
both components simultaneously and *smoothly*, producing a continuous arc
that approaches the origin without ever sliding along an axis.

This version guarantees the trajectory stays entirely in the first quadrant
by using a diagonal (axis-aligned) anisotropic Hessian. A diagonal A means
the two components decay independently as exponentials:
    theta_1(t) = theta_1(0) * exp(-lambda_1 * t)
    theta_2(t) = theta_2(0) * exp(-lambda_2 * t)
Both are strictly positive for all finite t, so the path never crosses
into negative territory.  Different lambda values (lambda_1 != lambda_2)
make the curve genuinely arc-shaped rather than a straight line to the origin.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "Polynomial-Regression-and-Model-Generalization_9.png"
)

AXIS_COLOR = "#1A1A1A"
PATH_COLOR = _style.PALETTE[1]    # red  (#EF4444)
INIT_COLOR = "#1F9BE0"            # steel-blue initial point


def make_trajectory(theta_init, lam1=2.5, lam2=0.6, n_pts=400):
    """Parametric continuous trajectory for diagonal ridge descent.

    theta_i(s) = theta_init_i * exp(-lambda_i * s),  s in [0, S].

    Both components are strictly positive for all s, so the curve never
    enters negative space.  The asymmetry (lam1 != lam2) causes theta_1 to
    decay faster, creating a genuinely curved (not straight) arc.

    Parameters
    ----------
    theta_init : array-like, shape (2,)
    lam1, lam2 : float  -- decay rates along theta_1 and theta_2 axes
    n_pts : int         -- number of sample points along the curve

    Returns
    -------
    xs, ys : ndarray of shape (n_pts,)
    """
    # Run long enough that both components are negligibly small (~1e-3 of init).
    S = max(np.log(theta_init[0] / 1e-3) / lam1,
            np.log(theta_init[1] / 1e-3) / lam2)
    s = np.linspace(0.0, S, n_pts)
    xs = theta_init[0] * np.exp(-lam1 * s)
    ys = theta_init[1] * np.exp(-lam2 * s)
    return xs, ys


def main():
    # Starting point: upper-right first quadrant (well away from any axis).
    theta_init = np.array([2.8, 3.2])

    # Faster decay along theta_1 (lam1 > lam2) -> curve bows toward theta_2 axis
    # before settling toward origin: a clear, visually interesting arc.
    xs, ys = make_trajectory(theta_init, lam1=2.5, lam2=0.6, n_pts=600)

    # Sanity check: every point strictly non-negative.
    assert np.all(xs >= 0) and np.all(ys >= 0), "Trajectory left first quadrant!"

    fig, ax = _style.new_ax(figsize=(7.0, 5.5))

    # --- Clean abstract axes (no ticks, no grid, no regular spines) ----------
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # Axis limits: give room for labels beyond the arrow tips.
    x_left, x_right = -0.5, 4.2
    y_bottom, y_top  = -0.5, 4.6

    # Horizontal axis -> theta_1 (rightward arrow).
    ax.annotate(
        "", xy=(x_right, 0.0), xytext=(x_left, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOR, lw=1.8),
        zorder=2,
    )
    # Vertical axis -> theta_2 (upward arrow).
    ax.annotate(
        "", xy=(0.0, y_top), xytext=(0.0, y_bottom),
        arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOR, lw=1.8),
        zorder=2,
    )

    # --- Red smooth trajectory -----------------------------------------------
    # Plot the full curve first.
    ax.plot(xs, ys, color=PATH_COLOR, linewidth=2.6,
            solid_capstyle="round", zorder=4)

    # Arrowhead: draw a short annotate segment near the tail of the curve
    # pointing toward (0, 0) so the arrowhead appears TO land on the origin.
    # Use index that is close to but not at the end to get a well-defined direction.
    arrow_start_idx = -25          # slightly upstream from the origin end
    ax.annotate(
        "",
        xy=(0.0, 0.0),                         # arrowhead at origin
        xytext=(xs[arrow_start_idx], ys[arrow_start_idx]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=PATH_COLOR,
            lw=2.6,
            shrinkA=0.0,
            shrinkB=0.0,
        ),
        zorder=5,
    )

    # --- Blue filled circle at the starting point ----------------------------
    ax.scatter(
        [theta_init[0]], [theta_init[1]],
        s=240, c=INIT_COLOR, edgecolors="white", linewidths=1.6, zorder=6,
    )

    # --- Text labels ---------------------------------------------------------
    # Axis-end labels.
    ax.text(x_right + 0.08, -0.22, r"$\theta_1$",
            ha="left", va="top", fontsize=15, color=AXIS_COLOR)
    ax.text(0.18, y_top,      r"$\theta_2$",
            ha="left", va="top", fontsize=15, color=AXIS_COLOR)

    # Origin label.
    ax.scatter([0.0], [0.0], s=55, c=AXIS_COLOR, zorder=5)
    ax.text(-0.15, -0.20, r"$O$  (0, 0)",
            ha="right", va="top", fontsize=12, color=AXIS_COLOR)

    # Initial-point annotation (Chinese, rendered via CJK font in _style).
    ax.annotate(
        "初始点（随机 θ）",
        xy=(theta_init[0], theta_init[1]),
        xytext=(theta_init[0] + 0.3, theta_init[1] + 0.3),
        fontsize=12, color=INIT_COLOR, va="bottom", ha="left",
        arrowprops=dict(arrowstyle="-", color=INIT_COLOR, lw=0.8),
    )

    # Mid-curve label for the trajectory.
    mid = len(xs) // 4              # pick a visible point on the arc
    ax.text(xs[mid] + 0.15, ys[mid] + 0.15,
            "岭回归梯度下降轨迹（L2）",
            fontsize=10, color=PATH_COLOR, va="bottom", ha="left",
            style="italic")

    # --- Title ---------------------------------------------------------------
    ax.set_title("岭回归（L2 正则）梯度下降：参数空间光滑弧线收敛到原点",
                 fontsize=13, pad=14)

    ax.set_aspect("equal")
    ax.set_xlim(x_left - 0.1, x_right + 0.8)
    ax.set_ylim(y_bottom - 0.1, y_top + 0.4)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="png")
    print("saved:", OUT_PATH)
    print("theta_init = (%.3f, %.3f)" % (theta_init[0], theta_init[1]))
    print("path x-range: [%.4f, %.4f]" % (xs.min(), xs.max()))
    print("path y-range: [%.4f, %.4f]" % (ys.min(), ys.max()))


if __name__ == "__main__":
    main()
