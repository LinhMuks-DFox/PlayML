"""Redraw of SVM_12: Soft Margin SVM motivation — the shifted decision boundary.

Compared with SVM_11 (Hard Margin), the Soft Margin decision boundary here
is moved upward/left so that the one outlier blue point (bottom-right) is
misclassified (falls on the red side of the line). This illustrates that
allowing one margin violation — tolerating one mis-classification — gives a
much wider margin for the majority of the data and better generalisation.

Key differences vs SVM_11:
  - Same point coordinates (including the outlier blue point, bottom-right).
  - Decision line is shifted: it no longer hugs the outlier, but cuts above it.
  - An annotation arrow labels the outlier as "Outlier（离群点）".
  - No margin band is drawn; only the decision line, matching the original.

Style: arrow-only axes (no numeric ticks), large solid dots, unified palette.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_12.png"
)

RED  = _style.PALETTE[1]   # #EF4444 — positive class (upper-right cluster)
BLUE = _style.PALETTE[0]   # #2563EB — negative class (lower-left cluster)
AMBER = _style.PALETTE[3]  # #F59E0B — outlier border highlight


def main():
    # ------------------------------------------------------------------
    # Point coordinates — identical to SVM_11 so the two figures share
    # the same layout and only the decision line differs.
    # ------------------------------------------------------------------
    red = np.array([
        [5.6, 8.4],   # top-centre
        [7.6, 7.9],   # upper-right
        [8.8, 7.9],   # upper-right
        [6.6, 6.5],   # middle-right
        [8.8, 6.4],   # right
        [7.9, 4.7],   # lower-right
    ])

    # Regular blue cluster (lower-left) — these are correctly classified.
    blue_main = np.array([
        [2.6, 5.0],   # left-middle
        [1.4, 2.9],   # lower-left
        [2.7, 2.2],   # lower-left
        [4.4, 3.0],   # lower-middle
        [4.7, 2.0],   # lower-middle
        [4.2, 3.9],   # lower-middle (extra point to match original density)
    ])

    # Single outlier blue point — bottom-right.
    # In SVM_11 the boundary was dragged down to accommodate it (Hard Margin).
    # In SVM_12 the boundary ignores it; the outlier ends up on the red side.
    blue_outlier = np.array([8.9, 3.8])

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = _style.new_ax(figsize=(6.2, 5.0))

    # Red (positive) scatter
    ax.scatter(
        red[:, 0], red[:, 1],
        color=RED, s=300, edgecolors="white", linewidths=1.4, zorder=3,
    )

    # Blue (negative) main cluster
    ax.scatter(
        blue_main[:, 0], blue_main[:, 1],
        color=BLUE, s=300, edgecolors="white", linewidths=1.4, zorder=3,
    )

    # Outlier blue point — same colour as other blues but with an amber
    # border ring so the reader immediately identifies it as the special point.
    ax.scatter(
        blue_outlier[0], blue_outlier[1],
        color=BLUE, s=320, edgecolors=AMBER, linewidths=2.5, zorder=4,
    )

    # ------------------------------------------------------------------
    # Decision boundary — Soft Margin line (shifted vs SVM_11).
    #
    # The line is defined so that it cuts ABOVE the outlier blue point,
    # placing that point on the "red" (positive) side.  This mimics what
    # happens when C is small (soft margin): the classifier accepts the
    # one violation to achieve a better overall margin for the bulk data.
    #
    # Line equation (data coordinates): y = slope * x + intercept
    #   slope     ≈ -0.80   (same general direction as SVM_11)
    #   intercept ≈  9.5    (shifted up so y(8.9) ≈ 2.4 < 3.8 = outlier.y)
    #
    # This puts the outlier at y=3.8 > line-at-x=8.9 (≈2.4) so it lies
    # ABOVE the line — i.e. on the red (positive) side → misclassified.
    # ------------------------------------------------------------------
    x_lo, x_hi = 0.0, 10.6
    y_lo, y_hi = 0.0, 9.4

    slope     = -0.78
    intercept =  9.2

    x_line = np.array([x_lo, x_hi])
    y_line = slope * x_line + intercept

    ax.plot(
        x_line, y_line,
        color="#111111", linewidth=2.4, zorder=2,
        solid_capstyle="round",
    )

    # ------------------------------------------------------------------
    # Annotation: arrow pointing to the outlier with Chinese label
    # ------------------------------------------------------------------
    ax.annotate(
        "Outlier（离群点）\n被容忍误分",
        xy=(blue_outlier[0], blue_outlier[1]),
        xytext=(6.6, 1.6),
        color="#1F2937",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="top",
        arrowprops=dict(
            arrowstyle="-|>",
            color="#6B7280",
            linewidth=1.5,
            mutation_scale=15,
            connectionstyle="arc3,rad=0.25",
        ),
        zorder=5,
    )

    # ------------------------------------------------------------------
    # Axes: hand-sketch arrow style — no numeric ticks, arrow ends only
    # ------------------------------------------------------------------
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    arrow_kw = dict(
        arrowstyle="-|>", color="#333333",
        linewidth=1.6, mutation_scale=18,
    )
    ax.annotate("", xy=(x_hi, y_lo), xytext=(x_lo, y_lo),
                arrowprops=arrow_kw, zorder=1)
    ax.annotate("", xy=(x_lo, y_hi), xytext=(x_lo, y_lo),
                arrowprops=arrow_kw, zorder=1)

    # Disable default grid (distracting for a concept sketch).
    ax.grid(False)

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ax.set_title("Soft Margin SVM：容忍离群点，决策线整体上移")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    fig.tight_layout()
    fig.savefig(OUT_PATH, format="png", dpi=200, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
