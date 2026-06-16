"""Redraw of Polynomial-Regression-and-Model-Generalization_10.

A conceptual diagram of how LASSO (L1) regularization moves the parameters
during gradient descent, in the 2D parameter plane (theta_1, theta_2).

Concept
-------
The L1 penalty alpha * |theta|_1 has gradient alpha * sign(theta); each
component is only -1 / 0 / +1, so the descent direction is constrained to be
axis-parallel. Starting from a point in the first quadrant (blue dot), the
path is therefore a *polyline*: it first travels diagonally (both components
shrinking by equal steps), hits a coordinate axis (one theta becomes exactly
0), and then slides straight down that axis to the origin. This "snap to an
axis" behavior is exactly why LASSO drives some coefficients to exactly 0
(feature selection) -- in contrast to Ridge (_9.png), whose path is a single
smooth curve that reaches the origin without zeroing any coordinate.

This is a pure schematic: there is no real data and no real gradient descent.
The polyline vertices are hand-specified to make the two qualitative features
obvious: (1) axis-parallel segments, and (2) hitting an axis (theta_1 = 0)
before the origin.

Styling (palette, CJK fonts) comes from the shared _style module. Because the
shared finalize() removes the top/right spines (a normal box plot), this
figure instead builds its own centered cross of arrowed axes via annotate and
hides every spine, then saves directly -- matching the original screenshot's
"abstract parameter plane" look.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "Polynomial-Regression-and-Model-Generalization_10.png"
)

DOT_COLOR = _style.PALETTE[0]    # blue  #2563EB  -- start point
PATH_COLOR = _style.PALETTE[1]   # red   #EF4444  -- descent polyline
AXIS_COLOR = "#333333"           # near-black abstract axes

# --- Schematic path vertices (first quadrant -> y axis -> origin) ----------
# Start in the first quadrant.
P_START = (3.6, 4.0)
# Diagonal axis-parallel-in-spirit move: equal shrink of both components
# until theta_1 reaches 0 (i.e. we hit the theta_2 axis / y axis).
P_CORNER = (0.0, 1.6)
# Then slide straight down the axis to the origin.
P_ORIGIN = (0.0, 0.0)

# Axis half-extents (abstract, no ticks).
X_LIM = (-2.0, 6.2)
Y_LIM = (-2.2, 6.2)


def _arrowed_axes(ax):
    """Draw a centered cross of two arrowed axes through the origin."""
    # Horizontal axis: from the left edge to a tip past the right.
    ax.annotate(
        "",
        xy=(X_LIM[1], 0.0),
        xytext=(X_LIM[0], 0.0),
        arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOR, lw=1.6,
                        shrinkA=0, shrinkB=0),
        zorder=1,
    )
    # Vertical axis: from the bottom edge to a tip past the top.
    ax.annotate(
        "",
        xy=(0.0, Y_LIM[1]),
        xytext=(0.0, Y_LIM[0]),
        arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOR, lw=1.6,
                        shrinkA=0, shrinkB=0),
        zorder=1,
    )


def _path_segment(ax, p_from, p_to):
    """One red arrowed segment of the descent polyline."""
    ax.annotate(
        "",
        xy=p_to,
        xytext=p_from,
        arrowprops=dict(arrowstyle="-|>", color=PATH_COLOR, lw=3.0,
                        shrinkA=0, shrinkB=0,
                        mutation_scale=22),
        zorder=3,
    )


def main():
    fig, ax = _style.new_ax(figsize=(6.4, 6.0))

    # Abstract plane: no grid, no ticks, equal aspect, all spines hidden.
    ax.grid(False)
    ax.set_aspect("equal")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Centered arrowed axes.
    _arrowed_axes(ax)

    # Axis-end labels theta_1 (horizontal) / theta_2 (vertical).
    ax.text(X_LIM[1] - 0.05, 0.0, r"$\theta_1$", color=AXIS_COLOR,
            fontsize=15, ha="right", va="top")
    ax.text(0.0, Y_LIM[1] - 0.05, r"$\theta_2$", color=AXIS_COLOR,
            fontsize=15, ha="left", va="top")

    # Descent polyline: diagonal -> hit the theta_2 axis -> down to origin.
    _path_segment(ax, P_START, P_CORNER)
    _path_segment(ax, P_CORNER, P_ORIGIN)

    # Start point: large blue dot in the first quadrant, on top.
    ax.scatter(*P_START, s=260, color=DOT_COLOR, edgecolors="white",
               linewidths=1.4, zorder=5)
    ax.annotate("起点", xy=P_START, xytext=(P_START[0] + 0.45, P_START[1] + 0.25),
                color=DOT_COLOR, fontsize=12, va="center")

    # Mark where the path hits the axis (one coefficient becomes exactly 0).
    ax.scatter(*P_CORNER, s=42, color=PATH_COLOR, zorder=5)
    ax.annotate(
        r"先碰到坐标轴：$\theta_1 = 0$",
        xy=P_CORNER, xytext=(P_CORNER[0] + 0.55, P_CORNER[1] + 0.55),
        color=PATH_COLOR, fontsize=11.5, va="center",
        arrowprops=dict(arrowstyle="->", color=PATH_COLOR, lw=1.1),
    )

    # Key-concept caption at the bottom.
    ax.text(
        0.5, -0.02,
        "LASSO：沿坐标轴折线下降，部分系数被精确置为 0（特征选择）",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=11.5, color="#333333",
    )

    ax.set_title("LASSO (L1) 正则化的参数下降路径", pad=14)

    # NOTE: cannot use _style.finalize here -- it re-shows nothing but is meant
    # for boxed axes. We saved directly to keep the centered-arrow look intact.
    fig.tight_layout()
    fig.savefig(OUT_PATH, format="png", dpi=200, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
