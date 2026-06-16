"""Redraw of PCAAndGradientAscent_06: PCA demean (zero-mean) concept figure.

Shows demeaned 2D data points (centroid at origin) with the first principal
component axis w drawn as a red arrow through the origin along ~45 degrees.
Dashed perpendicular lines drop from ALL data points to the PC axis,
illustrating orthogonal projection. The w label is placed near the arrow tip.

Style follows the unified _style.py (CJK-capable fonts, clean modern look).

Fixes (3rd attempt):
1. Data points are explicitly hand-placed along the 45-degree diagonal with
   SMALL perpendicular offsets, so the cloud is visually elongated along w.
   No SVD guessing — the PC axis is analytically set to exactly 45 degrees.
2. ALL 4 data points have dashed perpendicular projection lines to the PC axis.
3. Each dashed line is computed as the true perpendicular foot:
   foot = dot(p, u) * u, so the segment is guaranteed perpendicular to w.
4. The perpendicularity is visually obvious because the data spread perpendicular
   to w is small, making the projection drops short and clearly orthogonal.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "PCAAndGradientAscent_06.png"
)

POINT_COLOR = _style.PALETTE[0]  # blue  #2563EB
AXIS_COLOR = "#222222"           # near-black cross axes
PC_COLOR = _style.PALETTE[1]     # red   #EF4444 principal-component axis


# Fixed unit vector for the PC axis: exactly 45 degrees.
W_UNIT = np.array([1.0, 1.0]) / np.sqrt(2)
W_PERP = np.array([-1.0, 1.0]) / np.sqrt(2)   # 90-degree CCW rotation of W_UNIT


def make_data():
    """Design 4 demeaned 2D points clearly spread along the 45-degree diagonal.

    Strategy: explicitly place points along w with small perpendicular offsets.
    Diagonal positions (along w): -2.2, -0.7, 0.9, 2.0
    Perpendicular offsets: small values (+/-0.25) so the cloud is thin along perp.
    Enforce exact zero mean after construction.

    The key constraint: ALL perpendicular offsets must be small relative to the
    diagonal spread so the data visually looks elongated along w.
    """
    # Diagonal coords along w (the 45-degree direction):
    diag_coords = np.array([-2.2, -0.5, 0.8, 2.0])
    # Perpendicular offsets: large enough to make dashed lines clearly visible,
    # but still much smaller than the diagonal spread to keep the cloud elongated.
    perp_offsets = np.array([0.45, -0.65, 0.55, -0.50])

    pts = np.outer(diag_coords, W_UNIT) + np.outer(perp_offsets, W_PERP)

    # Enforce exact zero mean so centroid is at origin.
    pts = pts - pts.mean(axis=0)
    return pts


def draw_cross_axes(ax, x_extent, y_extent):
    """Draw a centered black cross (x/y axes) with arrowheads via annotate."""
    arrow = dict(arrowstyle="-|>", color=AXIS_COLOR, linewidth=2.0,
                 mutation_scale=20)
    ax.annotate("", xy=(x_extent, 0.0), xytext=(-x_extent, 0.0),
                arrowprops=arrow)
    ax.annotate("", xy=(0.0, y_extent), xytext=(0.0, -y_extent),
                arrowprops=arrow)

    # Chinese axis labels.
    ax.text(x_extent * 0.98, -0.12 * y_extent, "特征1", color=AXIS_COLOR,
            fontsize=13, ha="right", va="top")
    ax.text(0.07 * x_extent, y_extent * 0.98, "特征2", color=AXIS_COLOR,
            fontsize=13, ha="left", va="top")


def draw_pc_axis(ax, x_extent, y_extent):
    """Draw the red PC axis through the origin: line both ways + arrow at tip.

    The line is exactly 45 degrees (W_UNIT direction).
    Extends to the axis frame boundary so it visibly crosses the scatter cloud.
    """
    w = W_UNIT

    def t_to_edge(direction):
        """Max t such that t*direction stays inside the frame."""
        tx = x_extent / abs(direction[0]) if direction[0] != 0 else np.inf
        ty = y_extent / abs(direction[1]) if direction[1] != 0 else np.inf
        return min(tx, ty) * 0.92  # small margin so arrow tip is inside frame

    t_pos = t_to_edge(w)
    t_neg = t_to_edge(-w)

    p_pos = w * t_pos
    p_neg = -w * t_neg

    # Full line from negative to positive end.
    ax.plot([p_neg[0], p_pos[0]], [p_neg[1], p_pos[1]],
            color=PC_COLOR, linewidth=2.5, zorder=2, solid_capstyle="round")

    # Arrowhead at the positive tip only.
    ax.annotate(
        "",
        xy=(p_pos[0], p_pos[1]),
        xytext=(p_pos[0] - w[0] * 0.3, p_pos[1] - w[1] * 0.3),
        arrowprops=dict(arrowstyle="-|>", color=PC_COLOR, linewidth=2.5,
                        mutation_scale=22),
        zorder=3,
    )

    # Label 'w' placed just beyond the arrowhead tip, slightly off the line
    # in the perpendicular direction to avoid overlap.
    perp = W_PERP  # 90-degree CCW rotation of w
    label_pos = p_pos + w * 0.18 + perp * 0.25
    ax.text(label_pos[0], label_pos[1], "$w$",
            color=PC_COLOR, fontsize=16, ha="center", va="center",
            fontstyle="italic", fontweight="bold", zorder=5)


def draw_projections(ax, X):
    """Dashed perpendicular lines from ALL points to the PC axis.

    The foot of the perpendicular from point p onto the PC line (unit vector u):
        foot = dot(p, u) * u
    The dashed segment goes from p to foot — guaranteed perpendicular to w.
    This is applied to ALL data points.
    """
    u = W_UNIT
    for i in range(len(X)):
        p = X[i]
        foot = np.dot(p, u) * u   # foot of perpendicular on the PC line
        ax.plot([p[0], foot[0]], [p[1], foot[1]],
                linestyle="--", color="#555555", linewidth=1.5,
                zorder=4, dash_capstyle="round")
        # Small dot at the projection foot to mark it clearly.
        ax.plot(foot[0], foot[1], "o", color=PC_COLOR,
                markersize=5, zorder=4, markeredgewidth=0)


def main():
    X = make_data()

    x_ext = 3.2
    y_ext = 3.0

    fig, ax = _style.new_ax(figsize=(7, 6))
    ax.grid(False)

    draw_cross_axes(ax, x_ext, y_ext)

    draw_pc_axis(ax, x_ext, y_ext)

    # Draw perpendicular dashes for ALL data points.
    draw_projections(ax, X)

    # Blue filled scatter points.
    ax.scatter(X[:, 0], X[:, 1], s=180, c=POINT_COLOR, marker="o",
               edgecolors="white", linewidths=1.5, zorder=5)

    ax.set_xlim(-x_ext * 1.1, x_ext * 1.1)
    ax.set_ylim(-y_ext * 1.15, y_ext * 1.15)
    ax.set_aspect("equal")

    # Hide default frame and tick marks for a clean schematic look.
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)

    # Debug: verify geometry
    print(f"PC direction w = {W_UNIT}  (angle = {np.degrees(np.arctan2(W_UNIT[1], W_UNIT[0])):.1f} deg)")
    print(f"Data points:\n{X}")
    print(f"Mean: {X.mean(axis=0)}")

    # Verify each dashed line is perpendicular to w: the vector (p - foot) dot w == 0
    u = W_UNIT
    for i, p in enumerate(X):
        foot = np.dot(p, u) * u
        diff = p - foot
        dot_check = np.dot(diff, u)
        print(f"  Point {i}: p={p}, foot={foot}, (p-foot)·w = {dot_check:.2e}  (should be ~0)")


if __name__ == "__main__":
    main()
