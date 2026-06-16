"""Redraw of PCAAndGradientAscent_08: orthogonal-projection concept diagram.

A single data point X^(i) = (X_1, X_2) is projected orthogonally onto the
principal-component direction vector w = (w_1, w_2). The foot of the
perpendicular is the projection point (X_pr1^(i), X_pr2^(i)). The figure gives
an intuitive picture of "mapping a sample onto the w direction" and of
extracting / removing the first principal-component component.

This is a pure concept diagram: no dataset and no function curve. All points
are hand-specified, and the projection point is computed algebraically
( X_pr = (X . w / w . w) * w ) so the dashed perpendicular is geometrically
exact (better than the hand-drawn original).

Style follows the unified _style.py (clean modern look). Because this is a
schematic, the axes (ticks, grid, frame) are turned off to match the original
whitespace look, and ``axis('equal')`` keeps the right angle visually square.
All annotations are mathtext, so no CJK font is needed here.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "PCAAndGradientAscent_08.png"
)

ARROW_COLOR = _style.PALETTE[1]   # red  #EF4444 (both vectors, like original)
POINT_COLOR = _style.PALETTE[0]   # blue #2563EB (the data point)

# --- Hand-specified geometry ----------------------------------------------
# Common origin for both arrows.
O = np.array([0.0, 0.0])
# Principal-component direction; points to the upper right.
W_DIR = np.array([1.0, 0.55])
# Draw the w arrow as a scaled multiple of the direction so the tip sits well
# inside the canvas.
W_TIP = O + 2.6 * W_DIR
# Data point clearly below the w line (lower right).
X = np.array([2.3, -0.35])


def project(point, origin, direction):
    """Orthogonal projection of ``point`` onto the line through ``origin``
    with direction ``direction``.

    Returns the foot of the perpendicular: origin + (v . d / d . d) * d.
    """
    v = point - origin
    d = direction
    t = np.dot(v, d) / np.dot(d, d)
    return origin + t * d


def right_angle_marker(ax, corner, dir_a, dir_b, size=0.16, **kwargs):
    """Draw a small square right-angle marker at ``corner``.

    ``dir_a`` / ``dir_b`` are the two (roughly orthogonal) directions of the
    lines meeting at the corner; they are normalized internally.
    """
    a = dir_a / np.linalg.norm(dir_a)
    b = dir_b / np.linalg.norm(dir_b)
    p0 = corner + size * a
    p1 = corner + size * a + size * b
    p2 = corner + size * b
    ax.plot(
        [p0[0], p1[0], p2[0]],
        [p0[1], p1[1], p2[1]],
        color="#333333",
        linewidth=1.3,
        solid_capstyle="round",
        zorder=4,
        **kwargs,
    )


def main():
    fig, ax = _style.new_ax(figsize=(6.6, 4.2))

    # Projection (foot of perpendicular) onto the w line.
    X_pr = project(X, O, W_DIR)

    # --- Two red arrows from the common origin ---------------------------
    arrow_kw = dict(
        arrowprops=dict(
            arrowstyle="-|>",
            color=ARROW_COLOR,
            linewidth=2.4,
            mutation_scale=22,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=2,
    )
    # Arrow toward w (upper right).
    ax.annotate("", xy=W_TIP, xytext=O, **arrow_kw)
    # Arrow toward the data point X^(i) (lower right).
    ax.annotate("", xy=X, xytext=O, **arrow_kw)

    # --- Dashed (dotted) perpendicular: data point -> its projection -----
    ax.plot(
        [X[0], X_pr[0]],
        [X[1], X_pr[1]],
        color="#111111",
        linestyle=":",
        linewidth=2.6,
        zorder=3,
    )

    # --- Blue data point marker ------------------------------------------
    ax.scatter(
        [X[0]],
        [X[1]],
        s=240,
        c=POINT_COLOR,
        marker="o",
        edgecolors="white",
        linewidths=1.6,
        zorder=5,
    )

    # --- Right-angle marker at the foot of the perpendicular -------------
    # One leg runs back along the w line (toward the origin), the other runs
    # from the foot toward the data point.
    right_angle_marker(
        ax,
        X_pr,
        dir_a=O - X_pr,      # back along the w line
        dir_b=X - X_pr,      # along the perpendicular toward X
        size=0.18,
    )

    # --- Mathtext annotations --------------------------------------------
    # w label near the arrow tip.
    ax.annotate(
        r"$w = (w_1,\ w_2)$",
        xy=W_TIP,
        xytext=(W_TIP[0] + 0.25, W_TIP[1] + 0.05),
        fontsize=18,
        va="center",
        ha="left",
    )
    # Data-point label near the blue dot.
    ax.annotate(
        r"$X^{(i)} = (X_1^{(i)},\ X_2^{(i)})$",
        xy=X,
        xytext=(X[0] + 0.25, X[1] - 0.05),
        fontsize=17,
        va="center",
        ha="left",
    )
    # Projection-point label above the foot of the perpendicular.
    ax.annotate(
        r"$(X_{pr1}^{(i)},\ X_{pr2}^{(i)})$",
        xy=X_pr,
        xytext=(X_pr[0] - 0.15, X_pr[1] + 0.55),
        fontsize=17,
        va="bottom",
        ha="center",
    )

    # --- Canvas / framing -------------------------------------------------
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(-0.4, 5.6)
    ax.set_ylim(-1.2, 2.4)
    ax.axis("off")

    # finalize() saves PNG; we don't want its tight_layout to clip the axis,
    # but it is harmless with axis off. Keep using it for a single output path.
    fig.savefig(OUT_PATH, format="png", dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
