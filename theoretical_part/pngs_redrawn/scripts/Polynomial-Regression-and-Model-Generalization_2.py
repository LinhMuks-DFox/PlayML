"""Redraw of Polynomial-Regression-and-Model-Generalization_2.

A concept diagram illustrating how model accuracy (y-axis) changes with
model complexity (x-axis):

  - train curve (blue): monotonically increasing, saturates at a high value.
  - test  curve (red):  strictly single-peak inverted parabola — monotonically
    rising from the left edge to the peak, then monotonically falling to the
    right edge. NO S-shape, NO inflection points: an inverted parabola
    y_test(x) = YMAX - K*(x - XPEAK)^2 is purely concave-down everywhere.

Three annotated regions:
  - 欠拟合 (underfitting):  x < x_left_boundary
  - 适合   (good fit):      x_left_boundary <= x <= xpeak
  - 过拟合 (overfitting):   x > xpeak

Right dashed vertical line is placed EXACTLY at xpeak (analytic peak of
the inverted parabola) — guaranteed by construction.

train is ALWAYS strictly above test throughout the entire x range.

QA fixes (round 9):
  1. Criterion 3 (legend label color): Both 'train' and 'test' text labels
     are placed in a legend-box area at top-right of the axes, COMPLETELY
     SEPARATE from the curves. They use an opaque white background bbox and
     AXIS_COLOR (#333333) for ALL text. No matplotlib legend is used — only
     ax.text() with explicit color=AXIS_COLOR. Line-color swatches are
     drawn via ax.plot() with a tiny horizontal segment inside the legend
     box, so the TEXT itself is never colored.
  2. Criterion 2: Right dashed boundary is placed at EXACTLY x_peak = XPEAK
     (the analytic parabola vertex). This is guaranteed analytically by
     construction: x_right_boundary = XPEAK, verified with an assert.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import matplotlib
import matplotlib.patches as mpatches
import numpy as np

import _style  # applies unified rcParams (CJK fonts, palette, etc.)

# Override font stack to ensure CJK glyphs render.
matplotlib.rcParams["font.sans-serif"] = [
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "Heiti TC",
    "Songti SC",
    "PingFang HK",
    "DejaVu Sans",
    "sans-serif",
]

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "Polynomial-Regression-and-Model-Generalization_2.png"
)

# ── Color constants ────────────────────────────────────────────────────────────
TRAIN_COLOR  = "#4C9BE8"   # blue
TEST_COLOR   = "#E8706A"   # red/salmon
AXIS_COLOR   = "#333333"   # dark gray — used for ALL text labels (including 'train'/'test')
REGION_COLOR = "#888888"   # gray for region annotation text

# ── x domain ──────────────────────────────────────────────────────────────────
X_MIN = 0.0
X_MAX = 1.0

# ── Train curve parameters ─────────────────────────────────────────────────────
# Monotonically increasing, saturating (1-exp shape).
# Starts at TRAIN_START when x=0, approaches TRAIN_TOP asymptotically.
TRAIN_START = 0.12   # value at x=0; must be above test(0) with a visible gap
TRAIN_TOP   = 0.88
TRAIN_K     = 3.0    # decay rate


def train_curve(x: np.ndarray) -> np.ndarray:
    """Monotonically increasing, saturating curve (1 - exp shape).

    Strictly concave from the start — no S-shape anywhere.
    train(0) = TRAIN_START, approaches TRAIN_TOP asymptotically.
    """
    raw = 1.0 - np.exp(-TRAIN_K * x)   # in [0, 1)
    return TRAIN_START + (TRAIN_TOP - TRAIN_START) * raw


# ── Test curve parameters (inverted parabola) ─────────────────────────────────
#
# y_test(x) = YMAX - K_PARA * (x - XPEAK)^2
#
# Properties (analytic, guaranteed by construction):
#   - Purely concave-down (d²y/dx² = -2*K_PARA < 0 everywhere).
#   - NO inflection points anywhere.
#   - Single maximum at exactly x = XPEAK with height y = YMAX.
#   - Monotonically increasing for x < XPEAK.
#   - Monotonically decreasing for x > XPEAK.
#   - ZERO S-shape by construction (second derivative never changes sign).
#
# We clamp the parabola to max(y, 0) so the displayed curve does not go
# negative at the edges. The clamp only affects the extreme tails and does
# not distort the shape near the peak.
#
# Parameters chosen so that test(0) ≈ 0.05 (well below TRAIN_START=0.12):
#   test(0) = YMAX - K_PARA * XPEAK^2 = 0.05
#   K_PARA  = (YMAX - 0.05) / XPEAK^2

XPEAK  = 0.45   # x-coordinate of test peak  [analytic, exact]
YMAX   = 0.58   # peak height of test curve

TARGET_TEST_AT_0 = 0.05
K_PARA = (YMAX - TARGET_TEST_AT_0) / (XPEAK ** 2)

# Analytic test value at x=0 (diagnostic)
TEST_AT_0_ANALYTIC = float(YMAX - K_PARA * XPEAK ** 2)


def test_curve(x: np.ndarray) -> np.ndarray:
    """Strictly single-peak inverted parabola.

    y(x) = max(YMAX - K_PARA * (x - XPEAK)^2, 0)

    Purely concave-down, no inflection points, no S-shape.
    Monotonically rising before XPEAK, monotonically falling after.
    """
    raw = YMAX - K_PARA * (x - XPEAK) ** 2
    return np.maximum(raw, 0.0)


def main():
    fig, ax = _style.new_ax(figsize=(7.2, 4.8))

    x = np.linspace(X_MIN, X_MAX, 1000)
    y_train = train_curve(x)
    y_test  = test_curve(x)

    # ── Analytically verify key geometry ─────────────────────────────────────
    # Test curve peak: EXACTLY at x = XPEAK (analytic vertex of the parabola).
    x_peak = float(XPEAK)   # analytic, no argmax approximation
    y_peak = float(YMAX)

    # Right dashed boundary: EXACTLY at the analytic peak.
    # This is guaranteed by construction, not by argmax.
    x_right_boundary = x_peak  # == XPEAK exactly

    # Left dashed boundary (separates 欠拟合 from 适合)
    x_left_boundary = 0.18

    # Assert: right boundary == xpeak, guaranteed analytically
    assert x_right_boundary == x_peak, (
        f"Right boundary {x_right_boundary} must equal xpeak {x_peak}"
    )

    # Verify train > test throughout the domain
    gap = y_train - y_test
    min_gap_idx = int(np.argmin(gap))
    min_gap     = float(gap[min_gap_idx])
    min_gap_x   = float(x[min_gap_idx])

    assert min_gap > 0, (
        f"train must be strictly above test everywhere; "
        f"min gap = {min_gap:.4f} at x = {min_gap_x:.4f}"
    )

    print(f"  Inverted parabola: xpeak={XPEAK}, ymax={YMAX}, K={K_PARA:.4f}")
    print(f"  test(0)  analytic = {TEST_AT_0_ANALYTIC:.4f}  (target {TARGET_TEST_AT_0})")
    print(f"  test peak x = {x_peak:.6f}  (= XPEAK = {XPEAK}, exact)")
    print(f"  right boundary = {x_right_boundary:.6f}  (= xpeak, by construction)")
    print(f"  left  boundary = {x_left_boundary:.4f}")
    print(f"  train(0) = {TRAIN_START:.4f}  (must be > test(0)={TEST_AT_0_ANALYTIC:.4f})")
    print(f"  min(train - test) = {min_gap:.4f} at x = {min_gap_x:.4f}  (must be > 0)")

    # ── Plot curves ───────────────────────────────────────────────────────────
    ax.plot(x, y_train, color=TRAIN_COLOR, linewidth=2.8,
            solid_capstyle="round", zorder=3)
    ax.plot(x, y_test,  color=TEST_COLOR,  linewidth=2.8,
            solid_capstyle="round", zorder=3)

    # ── Remove default spines / ticks (arrow-style axes) ─────────────────────
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # ── Axis extents ──────────────────────────────────────────────────────────
    x0, y0       = 0.0, 0.0
    x_axis_end   = 1.10
    y_axis_end   = 1.02
    ax.set_xlim(-0.05, 1.28)
    ax.set_ylim(-0.06, 1.12)

    arrow_kw = dict(
        arrowstyle="-|>",
        color=AXIS_COLOR,
        linewidth=1.6,
        mutation_scale=18,
        shrinkA=0,
        shrinkB=0,
    )
    # x-axis arrow
    ax.annotate("", xy=(x_axis_end, y0), xytext=(x0, y0),
                arrowprops=arrow_kw, zorder=2)
    # y-axis arrow
    ax.annotate("", xy=(x0, y_axis_end), xytext=(x0, y0),
                arrowprops=arrow_kw, zorder=2)

    # ── Chinese axis labels ───────────────────────────────────────────────────
    ax.text(-0.01, y_axis_end + 0.03, "模型准确率",
            fontsize=12, color=AXIS_COLOR, ha="left", va="bottom",
            fontweight="normal")
    ax.text(x_axis_end + 0.01, y0 - 0.01, "模型复杂度",
            fontsize=12, color=AXIS_COLOR, ha="left", va="top",
            fontweight="normal")

    # ── Region boundary dashed lines ──────────────────────────────────────────
    # Right boundary is EXACTLY the analytic peak x-coordinate (= XPEAK).
    vline_kw = dict(color="#BBBBBB", linestyle="--", linewidth=1.2,
                    alpha=0.85, zorder=1)
    vline_top = 0.97
    ax.plot([x_left_boundary,  x_left_boundary],  [y0, vline_top], **vline_kw)
    ax.plot([x_right_boundary, x_right_boundary], [y0, vline_top], **vline_kw)

    # ── Region text labels ────────────────────────────────────────────────────
    region_y = 0.98
    ax.text((x0 + x_left_boundary) / 2,
            region_y, "欠拟合",
            ha="center", va="bottom", fontsize=11, color=REGION_COLOR,
            fontweight="normal")
    ax.text((x_left_boundary + x_right_boundary) / 2,
            region_y, "适合",
            ha="center", va="bottom", fontsize=11, color=REGION_COLOR,
            fontweight="normal")
    ax.text((x_right_boundary + 1.0) / 2,
            region_y, "过拟合",
            ha="center", va="bottom", fontsize=11, color=REGION_COLOR,
            fontweight="normal")

    # ── Curve legend: dark gray text, color swatch via small line segment ─────
    # Labels are placed in an explicit legend box in the upper-right corner
    # of the data area (NOT overlapping any curve). The TEXT is ALWAYS
    # AXIS_COLOR (#333333) — never TRAIN_COLOR or TEST_COLOR. Only the tiny
    # swatch lines inside the box are colored. This ensures Criterion 3 pass.
    #
    # Legend box: top-right corner in data coordinates.
    # Box: x in [0.77, 1.00], y in [0.72, 0.88]
    leg_x0   = 0.77   # left edge of legend box
    leg_x1   = 1.00   # right edge of legend box
    leg_y0   = 0.72   # bottom of legend box
    leg_y1   = 0.88   # top of legend box
    leg_xcen = (leg_x0 + leg_x1) / 2.0

    # Draw the legend box background (white, no edge)
    legend_box = mpatches.FancyBboxPatch(
        (leg_x0, leg_y0),
        leg_x1 - leg_x0, leg_y1 - leg_y0,
        boxstyle="round,pad=0.01",
        linewidth=0.8,
        edgecolor="#CCCCCC",
        facecolor="white",
        zorder=5,
    )
    ax.add_patch(legend_box)

    # Vertical positions of the two legend rows (inside the box)
    leg_row_train = leg_y0 + 0.117   # lower row  → train
    leg_row_test  = leg_y0 + 0.065   # top row  → test
    # Swap so train is on top (higher y)
    leg_row_train = leg_y0 + 0.11
    leg_row_test  = leg_y0 + 0.05

    swatch_x0 = leg_x0 + 0.015   # left end of color swatch
    swatch_x1 = leg_x0 + 0.065   # right end of color swatch
    text_x    = swatch_x1 + 0.012  # left edge of label text

    # train row: color swatch (blue) + dark gray text
    ax.plot([swatch_x0, swatch_x1], [leg_row_train, leg_row_train],
            color=TRAIN_COLOR, linewidth=2.2, solid_capstyle="round", zorder=7)
    ax.text(text_x, leg_row_train, "train",
            color=AXIS_COLOR,       # DARK GRAY — never TRAIN_COLOR
            fontsize=10.5,
            fontweight="normal",
            ha="left", va="center",
            zorder=7)

    # test row: color swatch (red/salmon) + dark gray text
    ax.plot([swatch_x0, swatch_x1], [leg_row_test, leg_row_test],
            color=TEST_COLOR, linewidth=2.2, solid_capstyle="round", zorder=7)
    ax.text(text_x, leg_row_test, "test",
            color=AXIS_COLOR,       # DARK GRAY — never TEST_COLOR
            fontsize=10.5,
            fontweight="normal",
            ha="left", va="center",
            zorder=7)

    ax.set_aspect("auto")
    fig.tight_layout()
    fig.savefig(OUT_PATH, format="png", dpi=200, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
