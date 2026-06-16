"""Redraw of SVM_14: Soft Margin SVM concept diagram.

Correct layout (from outer to inner):
  - Hard Margin support planes (solid blue): w^T x + b = ±1  (outermost)
  - Soft Margin dashed lines (pink):         w_d^T x + b_d = ±1  (inside hard margin)
  - Decision boundary (solid blue):          w^T x + b = 0   (centre)

Buffer zone (pink shading) = region between soft-margin dashed lines and hard-margin
solid lines on each side. This represents where violations ζ > 0 are tolerated.

No real SVM model is fitted — purely analytic parallel lines.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_14.png"
)

# Palette references
BLUE  = _style.PALETTE[0]   # #2563EB
RED   = _style.PALETTE[1]   # #EF4444
PINK  = _style.PALETTE[6]   # #EC4899  (magenta/pink for soft-margin dashed lines)
SLATE = _style.PALETTE[7]   # #64748B  (annotations, arrows)

# ---------------------------------------------------------------------------
# Geometry: all five lines are  y = SLOPE * x + intercept_k
#
# From outside to inside (top half):
#   Hard +1 line:  y = SLOPE*x + B_POS    (outermost solid)
#   Soft +1 line:  y = SLOPE*x + B_DPOS   (inner dashed, DELTA below B_POS)
#   Decision:      y = SLOPE*x + B_DB     (centre)
#   Soft -1 line:  y = SLOPE*x + B_DNEG   (inner dashed, DELTA above B_NEG)
#   Hard -1 line:  y = SLOPE*x + B_NEG    (outermost solid)
#
# Buffer zone = shading between (B_POS, B_DPOS) and between (B_NEG, B_DNEG)
# ---------------------------------------------------------------------------
SLOPE     = -0.9    # slope of all parallel lines
INTERCEPT = 5.0     # y-intercept of decision boundary
MARGIN    = 2.2     # separation (in y-intercept units) from DB to hard margin
DELTA     = 0.65    # inward shift from hard margin to soft-margin dashed line

B_DB   = INTERCEPT              # w^T x + b = 0
B_POS  = INTERCEPT + MARGIN     # Hard Margin +1
B_NEG  = INTERCEPT - MARGIN     # Hard Margin -1
B_DPOS = B_POS - DELTA          # Soft Margin +1  (inside hard +1)
B_DNEG = B_NEG + DELTA          # Soft Margin -1  (inside hard -1)


def line_y(x, intercept):
    """y = SLOPE * x + intercept  (all lines share same slope)."""
    return SLOPE * x + intercept


def perp_foot(x0, y0, intercept):
    """Foot of perpendicular from (x0, y0) onto line y = SLOPE*x + intercept."""
    m = SLOPE
    t = (x0 + m * (y0 - intercept)) / (1 + m * m)
    return t, m * t + intercept


def main():
    rng = np.random.default_rng(42)

    X_RANGE = (1.0, 9.0)

    # --- Synthetic data -------------------------------------------------------
    # Positive class: mostly above hard +1 line, a few between hard and soft lines
    n_pos_clear = 18
    n_pos_buf   = 4
    x_pc = rng.uniform(1.5, 8.5, n_pos_clear)
    y_pc = line_y(x_pc, B_POS) + rng.uniform(0.5, 2.5, n_pos_clear)
    x_pb = rng.uniform(2.0, 7.5, n_pos_buf)
    # place between soft +1 line and hard +1 line (buffer zone)
    y_pb = line_y(x_pb, B_DPOS) + rng.uniform(0.05, DELTA * 0.85, n_pos_buf)

    # Negative class: mostly below hard -1 line, a few between hard and soft lines
    n_neg_clear = 18
    n_neg_buf   = 4
    x_nc = rng.uniform(1.5, 8.5, n_neg_clear)
    y_nc = line_y(x_nc, B_NEG) - rng.uniform(0.5, 2.5, n_neg_clear)
    x_nb = rng.uniform(2.0, 7.5, n_neg_buf)
    # place between soft -1 line and hard -1 line (buffer zone)
    y_nb = line_y(x_nb, B_DNEG) - rng.uniform(0.05, DELTA * 0.85, n_neg_buf)

    x_pos = np.concatenate([x_pc, x_pb])
    y_pos = np.concatenate([y_pc, y_pb])
    x_neg = np.concatenate([x_nc, x_nb])
    y_neg = np.concatenate([y_nc, y_nb])

    # --- Figure ---------------------------------------------------------------
    fig, ax = _style.new_ax(figsize=(8.5, 6.2))

    x_plot = np.array([0.3, 9.8])

    # --- Buffer zone shading --------------------------------------------------
    # Two separate strips:
    #   Upper strip: between hard +1 (B_POS) and soft +1 (B_DPOS)
    #   Lower strip: between soft -1 (B_DNEG) and hard -1 (B_NEG)
    # This correctly shows the tolerance regions.
    y_hard_pos = line_y(x_plot, B_POS)
    y_soft_pos = line_y(x_plot, B_DPOS)
    y_soft_neg = line_y(x_plot, B_DNEG)
    y_hard_neg = line_y(x_plot, B_NEG)

    ax.fill_between(x_plot, y_soft_pos, y_hard_pos,
                    color=PINK, alpha=0.18, zorder=0)
    ax.fill_between(x_plot, y_hard_neg, y_soft_neg,
                    color=PINK, alpha=0.18, zorder=0)

    # --- Five parallel lines --------------------------------------------------
    # Hard Margin support planes (solid, blue) — outermost
    ax.plot(x_plot, line_y(x_plot, B_POS), color=BLUE,
            linewidth=1.9, linestyle="-", zorder=2)
    ax.plot(x_plot, line_y(x_plot, B_NEG), color=BLUE,
            linewidth=1.9, linestyle="-", zorder=2)

    # Soft Margin dashed lines (pink) — inside the hard margins
    ax.plot(x_plot, line_y(x_plot, B_DPOS), color=PINK,
            linewidth=1.8, linestyle="--", zorder=2)
    ax.plot(x_plot, line_y(x_plot, B_DNEG), color=PINK,
            linewidth=1.8, linestyle="--", zorder=2)

    # Decision boundary (solid, blue, slightly thicker) — centre
    ax.plot(x_plot, line_y(x_plot, B_DB), color=BLUE,
            linewidth=2.4, linestyle="-", zorder=2)

    # --- Scatter plots --------------------------------------------------------
    ax.scatter(x_pos, y_pos, color=RED,  s=60, edgecolors="white",
               linewidths=0.8, zorder=4, label="正类")
    ax.scatter(x_neg, y_neg, color=BLUE, s=60, edgecolors="white",
               linewidths=0.8, zorder=4, label="负类")

    # --- Landmark points A, B, C ---------------------------------------------
    # A: on the negative hard-margin support plane
    # B: foot of perpendicular from A on the decision boundary
    # C: foot of perpendicular from A on the positive hard-margin plane
    x_A = 2.8
    y_A = line_y(x_A, B_NEG)

    x_B, y_B = perp_foot(x_A, y_A, B_DB)
    x_C, y_C = perp_foot(x_A, y_A, B_POS)

    for (px, py, lbl, col, dx, dy) in [
        (x_A, y_A, "A", BLUE,  -0.15,  0.00),
        (x_B, y_B, "B", BLUE,  -0.15,  0.12),
        (x_C, y_C, "C", RED,   -0.15,  0.12),
    ]:
        ax.scatter([px], [py], color=col, s=90, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.text(px + dx, py + dy, lbl,
                fontsize=13, fontweight="bold", color=col,
                ha="right", va="bottom", zorder=6)

    # Perpendicular connector A -> C
    ax.plot([x_A, x_C], [y_A, y_C], color=SLATE, linewidth=1.0,
            linestyle=":", zorder=3)

    # --- Double-headed arrow for 'd' (from DB to +1 hard line) ---------------
    x_arr = x_B + 0.55
    y_arr_top = line_y(x_arr, B_POS)
    y_arr_bot = line_y(x_arr, B_DB)

    ax.annotate("", xy=(x_arr, y_arr_top),
                xytext=(x_arr, y_arr_bot),
                arrowprops=dict(arrowstyle="<->", color=SLATE,
                                linewidth=1.4, mutation_scale=12))
    ax.text(x_arr + 0.15, (y_arr_top + y_arr_bot) / 2,
            r"$d$", fontsize=13, color=SLATE, va="center", zorder=6)

    # --- '缓冲区' annotation pointing to the upper buffer strip ---------------
    # Tip: centre of upper buffer strip at x=6.5
    x_tip  = 6.5
    y_tip  = (line_y(x_tip, B_POS) + line_y(x_tip, B_DPOS)) / 2
    x_txt  = 7.4
    y_txt  = y_tip + 1.8

    ax.annotate(
        "缓冲区",
        xy=(x_tip, y_tip),
        xytext=(x_txt, y_txt),
        fontsize=12, color="#1F2937", ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-|>", color=SLATE,
                        linewidth=1.4, mutation_scale=14,
                        connectionstyle="arc3,rad=-0.25"),
        zorder=6,
    )

    # --- Formula labels on the right side (top to bottom visual order) --------
    # From top to bottom on the plot (largest y first at x_label):
    # B_POS  -> Hard +1
    # B_DPOS -> Soft +1
    # B_DB   -> centre
    # B_DNEG -> Soft -1
    # B_NEG  -> Hard -1
    x_label = 9.3
    label_specs = [
        (B_POS,  r"$\mathbf{w}^T\mathbf{x}+b=+1$",       BLUE),
        (B_DPOS, r"$\mathbf{w}_d^T\mathbf{x}+b_d=+1$",   PINK),
        (B_DB,   r"$\mathbf{w}^T\mathbf{x}+b=0$",         BLUE),
        (B_DNEG, r"$\mathbf{w}_d^T\mathbf{x}+b_d=-1$",   PINK),
        (B_NEG,  r"$\mathbf{w}^T\mathbf{x}+b=-1$",        BLUE),
    ]
    for b_val, tex, col in label_specs:
        y_at = line_y(x_label, b_val)
        ax.text(x_label + 0.05, y_at, tex,
                fontsize=9, color=col, va="center", ha="left", zorder=6)

    # --- Axes styling ---------------------------------------------------------
    ax.set_xlim(0.3, 11.5)
    y_all = np.concatenate([y_pos, y_neg])
    ax.set_ylim(y_all.min() - 1.2, y_all.max() + 2.8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title("Soft Margin SVM 概念示意", fontsize=14)
    ax.grid(False)

    # --- Legend ---------------------------------------------------------------
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor=RED, markeredgecolor="white", label="正类"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor=BLUE, markeredgecolor="white", label="负类"),
        Line2D([0], [0], color=BLUE, linewidth=2.0, linestyle="-",
               label=r"Hard Margin 边界"),
        Line2D([0], [0], color=PINK, linewidth=1.8, linestyle="--",
               label=r"Soft Margin 边界"),
        mpatches.Patch(facecolor=PINK, alpha=0.35, label="缓冲区"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=9, framealpha=0.9)

    # --- Save -----------------------------------------------------------------
    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
