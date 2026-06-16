"""Redraw of PCAAndGradientAscent_05: PCA concept sketch (before demean).

A small set of 2D samples (特征1, 特征2) scattered roughly along a diagonal
line. A red, upward-sloping arrow axis -- a *candidate principal component
direction* -- runs through the cloud. From every sample a short dotted foot
of perpendicular drops onto that axis, picturing the orthogonal projection of
the sample onto the candidate axis. This expresses the PCA idea of "find an
axis that maximises the variance of the projected samples", matching the
notes' illustration *before* demeaning (so the cloud has a non-zero mean).

This is a pure teaching schematic, so we deviate from the unified grid/ticks:
- arrow-tipped left + bottom axes drawn with ``annotate`` (no box, no grid),
- no tick numbers, no legend (the original has none),
- blue dots (#2563EB) and a red candidate axis ('r'), matching the original.

Pairs with PCAAndGradientAscent_06 (after demean): same cloud shape, but here
the mean is deliberately non-zero to show the "not yet demeaned" state.

Style still comes from the unified _style.py (CJK-capable fonts); we just
disable the grid/spines locally before saving via _style.finalize.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "PCAAndGradientAscent_05.png"
)

POINT_COLOR = _style.PALETTE[0]  # blue #2563EB, matches the original dots
AXIS_COLOR = _style.PALETTE[1]   # red #EF4444, the candidate PC axis


def make_points():
    """A few (~6) demo samples scattered along a diagonal with mild noise.

    Mirrors the notebook idiom ``X[:,1] = 0.75*X[:,0] + 3 + noise`` but with a
    tiny, hand-seeded sample so the perpendicular feet are easy to read. The
    cloud sits in the positive quadrant with a non-zero mean (pre-demean).
    """
    x = np.array([1.6, 2.7, 3.6, 4.5, 5.3, 6.2])
    # Hand-picked vertical offsets so each point sits clearly *off* the
    # candidate axis (alternating above/below): this makes the dotted
    # feet-of-perpendicular long enough to read, like the original.
    offsets = np.array([-0.30, 0.45, -0.35, 0.55, 0.30, -0.45])
    y = 0.6 * x + 1.2 + offsets
    return np.column_stack([x, y])


def candidate_axis_direction():
    """Unit vector of the candidate principal component (positive slope)."""
    w = np.array([1.0, 0.6])           # slope ~0.6, matches the spec/original
    return w / np.linalg.norm(w)


def feet_of_perpendicular(points, anchor, direction):
    """Orthogonal projection of each point onto the line anchor + t*direction.

    Returns the foot-of-perpendicular point for every sample. ``direction``
    must be a unit vector.
    """
    rel = points - anchor
    t = rel @ direction                # scalar projection coordinate
    return anchor + np.outer(t, direction)


def main():
    fig, ax = _style.new_ax(figsize=(7.2, 4.6))

    points = make_points()
    direction = candidate_axis_direction()
    # Anchor the candidate line near the cloud's centre so the feet land on it.
    anchor = points.mean(axis=0)

    feet = feet_of_perpendicular(points, anchor, direction)

    # --- Red candidate axis, extended a bit past the cloud on both ends. ---
    t_vals = (points - anchor) @ direction
    t_lo, t_hi = t_vals.min() - 1.6, t_vals.max() + 1.6
    p_lo = anchor + t_lo * direction
    p_hi = anchor + t_hi * direction
    ax.annotate(
        "",
        xy=(p_hi[0], p_hi[1]),
        xytext=(p_lo[0], p_lo[1]),
        arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOR, lw=2.4,
                        shrinkA=0, shrinkB=0),
        zorder=2,
    )

    # --- Dotted feet of perpendicular from each sample onto the red axis. ---
    for (px, py), (fx, fy) in zip(points, feet):
        ax.plot([px, fx], [py, fy], linestyle=":", color="black",
                linewidth=1.4, zorder=3)

    # --- Blue samples. ---
    ax.scatter(points[:, 0], points[:, 1], s=160, c=POINT_COLOR, marker="o",
               edgecolors="white", linewidths=1.4, zorder=4)

    # --- Teaching coordinate frame: arrow axes, no ticks, no grid. ---
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    x0, y0 = 0.0, 0.0
    x_max = points[:, 0].max() + 1.4
    y_max = points[:, 1].max() + 1.2
    ax.set_xlim(-0.4, x_max)
    ax.set_ylim(-0.4, y_max)

    arrow_kw = dict(arrowstyle="-|>", color="#333333", lw=1.8,
                    shrinkA=0, shrinkB=0)
    # Bottom axis (特征1) and left axis (特征2), both arrow-tipped.
    ax.annotate("", xy=(x_max, y0), xytext=(x0, y0), arrowprops=arrow_kw,
                zorder=1)
    ax.annotate("", xy=(x0, y_max), xytext=(x0, y0), arrowprops=arrow_kw,
                zorder=1)

    # Axis labels at the arrow tips.
    ax.text(x_max, y0 - 0.12, "特征1", ha="right", va="top", fontsize=14)
    ax.text(x0 + 0.12, y_max, "特征2", ha="left", va="top", fontsize=14)

    ax.set_aspect("equal", adjustable="box")

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
