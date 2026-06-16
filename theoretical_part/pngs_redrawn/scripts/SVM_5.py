"""Redraw of SVM_5: 最近样本到决策边界的距离 (margin motivation for SVM).

Two classes of 2-D points are separated by a decision boundary (line).
Three support vectors (2 red, 1 blue) are the samples closest to the boundary.
Perpendicular distance segments from the SVs to the boundary illustrate
the concept "distance from the nearest sample to the decision boundary".

Geometric design (all coordinates strictly computed via numpy — no eye-balling):

Decision boundary:  y = -0.5 x + 7.5   (slope m = -0.5, through (5, 5))
Perpendicular slope: -1/m = +2.0  (positive, steep — goes from lower-left to upper-right)
Unit normal toward red (above-line) side:
    n_hat = (-m, 1) / |(-m,1)| = (0.5, 1.0) / sqrt(1.25) ≈ (0.4472, 0.8944)
    angle ≈ +63.4°

All three dashed segments are proven perpendicular:
    segment = sv - foot = ±d_sv * n_hat
    dot(segment, (1, m)) = ±d_sv * dot(n_hat, (1, m)) = 0  (by construction)

Support vectors placed at perp-dist d_sv = 1.2 from the line.
Background points placed at perp-dist >= clearance = 2.5 > d_sv.

Axes use set_aspect('equal') so mathematical perpendicularity matches visual
perpendicularity — the dashed lines appear at true slope +2.0 in the figure.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_5.png"

RED        = _style.PALETTE[1]   # #EF4444
BLUE       = _style.PALETTE[0]   # #2563EB
LINE_COLOR = "#333333"
SEG_COLOR  = "#111111"


# ---------------------------------------------------------------------------
# Geometry helpers (all computed analytically — no guessing)
# ---------------------------------------------------------------------------

def foot_of_perpendicular(px, py, slope, intercept):
    """Return foot of perpendicular from (px, py) onto y = slope*x + intercept.

    Rewrite line as:  slope*x - y + intercept = 0  =>  a=slope, b=-1, c=intercept
    Standard foot formula:
        t = (a*px + b*py + c) / (a² + b²)
        fx = px - a*t,   fy = py - b*t
    This satisfies: dot(foot - point, line_dir) = 0  exactly.
    """
    a, b_c, c = slope, -1.0, intercept
    denom = a * a + b_c * b_c          # slope² + 1
    t = (a * px + b_c * py + c) / denom
    return np.array([px - a * t, py - b_c * t])


def perp_dist(px, py, slope, intercept):
    """Unsigned perpendicular distance from (px,py) to y = slope*x + intercept."""
    return abs(slope * px - py + intercept) / np.sqrt(slope ** 2 + 1)


def line_y(x, slope, intercept):
    return slope * x + intercept


def main():
    # -----------------------------------------------------------------------
    # Coordinate frame
    # -----------------------------------------------------------------------
    x_lo, x_hi = 0.0, 10.0
    y_lo, y_hi = 0.0, 10.0

    # -----------------------------------------------------------------------
    # Decision boundary parameters
    # -----------------------------------------------------------------------
    slope     = -0.5          # line: y = -0.5 x + 7.5
    intercept =  7.5

    # Perpendicular slope (verified: -1 / slope = 2.0 > 0)
    slope_perp = -1.0 / slope  # = 2.0

    # Unit normal toward red (above-line) side.
    # "above" means y > slope*x + intercept, equivalently slope*x - y + intercept < 0.
    # The normal direction (a, -b) = (-slope, 1) = (0.5, 1.0) points toward +y side of line.
    n_raw = np.array([-slope, 1.0])      # (0.5, 1.0)
    n_hat = n_raw / np.linalg.norm(n_raw)

    print(f"Decision line: y = {slope}x + {intercept}")
    print(f"Perpendicular slope: {slope_perp:.4f}")
    print(f"n_hat = {n_hat}  magnitude = {np.linalg.norm(n_hat):.6f}")
    print(f"n_hat angle = {np.degrees(np.arctan2(n_hat[1], n_hat[0])):.2f}°")

    # -----------------------------------------------------------------------
    # Support vector geometry
    # -----------------------------------------------------------------------
    d_sv      = 1.2    # perp distance of each SV to the boundary
    clearance = 2.5    # minimum perp distance for ALL background points
    #                    gap = clearance - d_sv = 1.3  → SVs are truly nearest

    # Choose foot x-coordinates for the 3 SVs.
    # Constraints:
    #   • Foot must be on the drawn segment (x in [1, 9])
    #   • Displaced SV must stay within [x_lo+0.5, x_hi-0.5] × [y_lo+0.5, y_hi-0.5]
    #   • r1 left, r2 right of b1 (visually separated)
    foot_r1_x = 3.0    # left red SV foot
    foot_r2_x = 7.0    # right red SV foot
    foot_b1_x = 5.0    # blue SV foot (between the two red feet)

    foot_r1 = np.array([foot_r1_x, line_y(foot_r1_x, slope, intercept)])
    foot_r2 = np.array([foot_r2_x, line_y(foot_r2_x, slope, intercept)])
    foot_b1 = np.array([foot_b1_x, line_y(foot_b1_x, slope, intercept)])

    # Displace along ±n_hat to place SVs
    red_sv1 = foot_r1 + d_sv * n_hat    # above line (red class)
    red_sv2 = foot_r2 + d_sv * n_hat    # above line (red class)
    blue_sv = foot_b1 - d_sv * n_hat    # below line (blue class)

    for label, sv, foot in [
        ("red_sv1", red_sv1, foot_r1),
        ("red_sv2", red_sv2, foot_r2),
        ("blue_sv", blue_sv, foot_b1),
    ]:
        print(f"\n{label}: sv=({sv[0]:.4f}, {sv[1]:.4f}), foot=({foot[0]:.4f}, {foot[1]:.4f})")
        computed_foot = foot_of_perpendicular(sv[0], sv[1], slope, intercept)
        err_foot = np.linalg.norm(computed_foot - foot)
        assert err_foot < 1e-9, f"{label}: foot error={err_foot:.2e}"

        seg = sv - foot
        seg_angle = np.degrees(np.arctan2(seg[1], seg[0]))
        print(f"  segment angle (foot→sv): {seg_angle:.2f}°  "
              f"(expected ±{abs(np.degrees(np.arctan2(n_hat[1], n_hat[0]))):.2f}°)")

        line_dir = np.array([1.0, slope])
        dot = np.dot(seg, line_dir)
        assert abs(dot) < 1e-9, f"{label}: dot(seg, line_dir)={dot:.2e}  NOT perpendicular!"
        print(f"  dot(seg, line_dir)={dot:.2e}  ✓ perpendicular")

        d_check = perp_dist(sv[0], sv[1], slope, intercept)
        assert abs(d_check - d_sv) < 1e-9, f"{label}: d={d_check:.6f} ≠ {d_sv}"
        print(f"  perp_dist={d_check:.6f} = d_sv={d_sv}  ✓")

    # -----------------------------------------------------------------------
    # Background scatter
    # -----------------------------------------------------------------------
    # Strategy: directly specify visible (x, y) of each background point and
    # verify it satisfies the clearance constraint.  This avoids out-of-bounds
    # points caused by the -n_hat displacement mechanism.

    def check_bg(pts, class_side):
        """
        Verify a list of (x, y) background points.
        class_side: +1 for red (above line), -1 for blue (below line).
        Returns validated list.
        """
        valid = []
        for (bx, by) in pts:
            on_correct = ((by > line_y(bx, slope, intercept)) if class_side == 1
                          else (by < line_y(bx, slope, intercept)))
            d = perp_dist(bx, by, slope, intercept)
            in_box = (x_lo + 0.1 <= bx <= x_hi - 0.1 and
                      y_lo + 0.1 <= by <= y_hi - 0.1)
            if d >= clearance and on_correct and in_box:
                valid.append(np.array([bx, by]))
            else:
                print(f"  [dropped] ({bx:.2f},{by:.2f}): d={d:.3f}, "
                      f"side={'ok' if on_correct else 'WRONG'}, "
                      f"in_box={in_box}")
        return np.array(valid)

    # Red background: above the decision line (y > -0.5x + 7.5).
    # Points spread from x≈2 to x≈9, and high enough in y.
    # Decision line at x=2 is y=6.5, at x=9 is y=3.0.
    # Each point's distance from the line is at least clearance=2.5.
    red_bg_pts = [
        (3.2, 9.2),    # d ≈ 3.2
        (4.0, 9.5),    # d ≈ 3.6
        (5.0, 8.8),    # d ≈ 2.5 — just at clearance
        (5.5, 9.3),    # d ≈ 3.1
        (6.0, 8.8),    # d ≈ 2.9
        (6.8, 8.5),    # d ≈ 2.7
        (7.0, 9.0),    # d ≈ 3.2
        (7.5, 8.2),    # d ≈ 2.6
        (8.2, 7.8),    # d ≈ 2.7
        (9.0, 7.2),    # d ≈ 2.5
    ]

    # Blue background: below the decision line (y < -0.5x + 7.5).
    # Spread from x≈1.5 to x≈7, not all clustering in one corner.
    # Decision line at x=1.5 is y=6.75, at x=7 is y=4.0.
    # Each point's distance from the line is at least clearance=2.5.
    blue_bg_pts = [
        (1.5, 3.0),   # d = 3.35
        (2.0, 2.0),   # d = 4.02
        (2.5, 3.2),   # d = 2.73
        (3.0, 1.8),   # d = 3.76
        (3.5, 2.5),   # d = 2.91
        (4.5, 1.5),   # d = 3.35
        (5.0, 2.0),   # d = 2.68  ← just above clearance
        (5.5, 1.2),   # d = 3.18
        (6.0, 0.8),   # d = 3.40
        (6.5, 0.9),   # d = 3.09
    ]

    red_bg  = check_bg(red_bg_pts,  class_side= 1)
    blue_bg = check_bg(blue_bg_pts, class_side=-1)
    print(f"\nRed background: {len(red_bg)} points")
    print(f"Blue background: {len(blue_bg)} points")

    # -----------------------------------------------------------------------
    # Final constraint check: SVs must be nearest points in their class
    # -----------------------------------------------------------------------
    sv_d_r1 = perp_dist(red_sv1[0], red_sv1[1], slope, intercept)
    sv_d_r2 = perp_dist(red_sv2[0], red_sv2[1], slope, intercept)
    sv_d_b1 = perp_dist(blue_sv[0], blue_sv[1], slope, intercept)
    assert abs(sv_d_r1 - d_sv) < 1e-9
    assert abs(sv_d_r2 - d_sv) < 1e-9
    assert abs(sv_d_b1 - d_sv) < 1e-9
    print(f"\nSV distances: r1={sv_d_r1:.4f}, r2={sv_d_r2:.4f}, b1={sv_d_b1:.4f}  (all = {d_sv})")
    # Both sides have equal margin:
    print(f"Margin equality: red_side={sv_d_r1:.4f} == blue_side={sv_d_b1:.4f}  ✓")

    for i, pt in enumerate(red_bg):
        d = perp_dist(pt[0], pt[1], slope, intercept)
        assert d >= clearance, f"red_bg[{i}] d={d:.4f} < clearance={clearance}"
    for i, pt in enumerate(blue_bg):
        d = perp_dist(pt[0], pt[1], slope, intercept)
        assert d >= clearance, f"blue_bg[{i}] d={d:.4f} < clearance={clearance}"

    print("All geometry assertions passed.")

    # -----------------------------------------------------------------------
    # Draw
    # -----------------------------------------------------------------------
    # Use equal aspect so that mathematically perpendicular dashes also appear
    # visually perpendicular in the saved PNG.
    fig, ax = _style.new_ax(figsize=(6.4, 6.4))
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")   # ← critical: makes slope-2.0 dashes look like slope 2.0

    # --- Decision boundary ---------------------------------------------------
    bd_xs = np.linspace(0.0, 10.0, 300)
    bd_ys = slope * bd_xs + intercept
    ax.plot(bd_xs, bd_ys, color=LINE_COLOR, linewidth=2.4,
            solid_capstyle="round", zorder=2, label="_nolegend_")

    # --- Perpendicular distance segments (SV → foot on boundary) ------------
    # Segment direction = ±d_sv * n_hat, which is perpendicular to the line.
    # dot(n_hat, (1, slope)) = 0  (verified analytically above).
    for sv, foot in [(red_sv1, foot_r1), (red_sv2, foot_r2), (blue_sv, foot_b1)]:
        ax.plot([sv[0], foot[0]], [sv[1], foot[1]],
                color=SEG_COLOR, linewidth=1.8,
                linestyle="--", dash_capstyle="round",
                zorder=2.5)

    # --- Foot markers (small open circles on the decision line) -------------
    for foot in [foot_r1, foot_r2, foot_b1]:
        ax.scatter(foot[0], foot[1], color="white", s=36,
                   edgecolors=SEG_COLOR, linewidths=1.4, zorder=5)

    # --- Background scatter (ordinary points) --------------------------------
    if len(red_bg) > 0:
        ax.scatter(red_bg[:, 0], red_bg[:, 1],
                   color=RED, s=90, edgecolors="white", linewidths=0.8, zorder=3)
    if len(blue_bg) > 0:
        ax.scatter(blue_bg[:, 0], blue_bg[:, 1],
                   color=BLUE, s=90, edgecolors="white", linewidths=0.8, zorder=3)

    # --- Support vectors (larger markers with darker edges) ------------------
    sv_size   = 220
    sv_edge_r = "#7f1d1d"
    sv_edge_b = "#1e3a8a"

    for sv, color, edge in [
        (red_sv1, RED,  sv_edge_r),
        (red_sv2, RED,  sv_edge_r),
        (blue_sv, BLUE, sv_edge_b),
    ]:
        ax.scatter(sv[0], sv[1], color="white", s=sv_size + 100, zorder=4.0)
        ax.scatter(sv[0], sv[1], color=color, s=sv_size,
                   edgecolors=edge, linewidths=2.2, zorder=4.5)

    # --- Axis arrows (hand-sketch style) ------------------------------------
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    arrow_kw = dict(arrowstyle="-|>", color="#333333",
                    linewidth=1.4, mutation_scale=16)
    ax.annotate("", xy=(x_hi, y_lo), xytext=(x_lo, y_lo),
                arrowprops=arrow_kw, zorder=1)
    ax.annotate("", xy=(x_lo, y_hi), xytext=(x_lo, y_lo),
                arrowprops=arrow_kw, zorder=1)

    # --- Legend --------------------------------------------------------------
    bd_handle = mlines.Line2D(
        [], [], color=LINE_COLOR, linewidth=2.0,
        label="决策边界")
    seg_handle = mlines.Line2D(
        [], [], color=SEG_COLOR, linewidth=1.5, linestyle="--",
        label="点到决策边界的距离")
    sv_patch_r = mpatches.Patch(
        facecolor=RED,  edgecolor=sv_edge_r, linewidth=1.5,
        label="支撑向量（红色类）")
    sv_patch_b = mpatches.Patch(
        facecolor=BLUE, edgecolor=sv_edge_b, linewidth=1.5,
        label="支撑向量（蓝色类）")

    ax.legend(handles=[bd_handle, seg_handle, sv_patch_r, sv_patch_b],
              loc="lower right", fontsize=9, framealpha=0.92)

    # --- Title ---------------------------------------------------------------
    ax.set_title("最近样本到决策边界的距离", pad=8)
    ax.grid(False)

    # --- Save ----------------------------------------------------------------
    _style.finalize(fig, OUT_PATH)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
