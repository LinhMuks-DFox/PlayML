"""Redraw of Ensemble_3: Stacking ensemble architecture concept diagram.

Two-level stacking structure:
  - Bottom: input node (New instance)
  - Level 1: three base estimator boxes (blue/purple/pink)
  - Level 1.5: three prediction circles (3.1, 2.7, 2.9)
  - Level 2: blending model box (green)
  - Top: final output circle (3.0)

Pure concept diagram -- no data, no axes.
"""

import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')

import _style  # noqa: F401  (applies unified rcParams on import)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

OUT_PATH = '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Ensemble_3.png'

# ---------------------------------------------------------------------------
# Layout constants (all in data coords on a 10 x 12 canvas)
# ---------------------------------------------------------------------------
CANVAS_W = 10.0
CANVAS_H = 12.0

# Y levels (bottom to top)
Y_INPUT       = 1.0   # New instance node
Y_BASE        = 3.5   # Base estimator boxes  (center y)
Y_PRED        = 6.2   # Prediction circles
Y_BLEND       = 8.5   # Blending model box    (center y)
Y_OUTPUT      = 10.8  # Final output circle

# X positions for the three base estimators
X_BASES = [2.2, 5.0, 7.8]
X_CENTER = 5.0

# Node sizes
INPUT_RADIUS  = 0.45
PRED_RADIUS   = 0.45
OUTPUT_RADIUS = 0.45

# Box dimensions
BASE_W  = 1.5
BASE_H  = 1.1
BLEND_W = 2.0
BLEND_H = 1.1

# Colors
COL_BLUE   = '#AED6F1'
COL_PURPLE = '#D7BDE2'
COL_PINK   = '#F9B8C0'
COL_GREEN  = '#A9DFBF'
COL_BORDER = '#555555'
COL_ARROW  = '#888888'
COL_TEXT   = '#222222'
COL_INPUT_FILL = '#FDEBD0'


def draw_arrow(ax, x0, y0, x1, y1, shrinkA=6, shrinkB=6):
    """Draw a gray FancyArrowPatch from (x0,y0) to (x1,y1)."""
    arrow = FancyArrowPatch(
        posA=(x0, y0), posB=(x1, y1),
        arrowstyle='-|>',
        color=COL_ARROW,
        linewidth=1.5,
        mutation_scale=14,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        zorder=2,
    )
    ax.add_patch(arrow)


def draw_gear_symbol(ax, cx, cy, color='#555555', size=13):
    """Draw a simplified gear icon using concentric circles and wedges."""
    import matplotlib.patches as mp
    # Outer toothed ring -- approximated by a larger dashed circle
    outer = mp.Circle((cx, cy + 0.14), 0.22,
                       facecolor='none', edgecolor=color,
                       linewidth=2.0, linestyle='--', zorder=5)
    ax.add_patch(outer)
    # Inner hub circle
    inner = mp.Circle((cx, cy + 0.14), 0.10,
                       facecolor=color, edgecolor='none',
                       alpha=0.5, zorder=6)
    ax.add_patch(inner)
    # Four short lines as "teeth" at N/S/E/W
    tooth_len = 0.12
    for angle_deg in [0, 90, 180, 270]:
        angle_rad = np.radians(angle_deg)
        x0 = cx + 0.22 * np.cos(angle_rad)
        y0 = (cy + 0.14) + 0.22 * np.sin(angle_rad)
        x1 = cx + (0.22 + tooth_len) * np.cos(angle_rad)
        y1 = (cy + 0.14) + (0.22 + tooth_len) * np.sin(angle_rad)
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=2.5, zorder=6, solid_capstyle='round')


def draw_base_box(ax, cx, cy, facecolor, label_top, label_bot):
    """Draw a rounded rectangle base estimator box with gear icon."""
    x = cx - BASE_W / 2
    y = cy - BASE_H / 2
    box = FancyBboxPatch(
        (x, y), BASE_W, BASE_H,
        boxstyle='round,pad=0.08',
        facecolor=facecolor,
        edgecolor=COL_BORDER,
        linewidth=1.4,
        zorder=3,
    )
    ax.add_patch(box)
    # Gear icon (top part)
    draw_gear_symbol(ax, cx, cy, color='#444444', size=14)
    # Label text (bottom part)
    ax.text(
        cx, cy - 0.28,
        label_bot,
        ha='center', va='top',
        fontsize=8,
        color='#333333',
        zorder=5,
    )


def draw_blend_box(ax, cx, cy):
    """Draw the blending model rounded rectangle."""
    x = cx - BLEND_W / 2
    y = cy - BLEND_H / 2
    box = FancyBboxPatch(
        (x, y), BLEND_W, BLEND_H,
        boxstyle='round,pad=0.08',
        facecolor=COL_GREEN,
        edgecolor=COL_BORDER,
        linewidth=1.6,
        zorder=3,
    )
    ax.add_patch(box)
    draw_gear_symbol(ax, cx, cy, color='#1a6b3a', size=16)
    ax.text(
        cx, cy - 0.3,
        'Blending',
        ha='center', va='top',
        fontsize=9,
        color='#1a6b3a',
        fontweight='bold',
        zorder=5,
    )


def draw_circle_node(ax, cx, cy, radius, facecolor, text, fontsize=12, fontweight='bold'):
    """Draw a filled circle with centered text."""
    circ = Circle(
        (cx, cy), radius,
        facecolor=facecolor,
        edgecolor=COL_BORDER,
        linewidth=1.5,
        zorder=4,
    )
    ax.add_patch(circ)
    ax.text(
        cx, cy,
        text,
        ha='center', va='center',
        fontsize=fontsize,
        color=COL_TEXT,
        fontweight=fontweight,
        zorder=5,
    )


def draw_input_node(ax, cx, cy):
    """Draw the input node: circle with otimes (x in circle) symbol."""
    circ = Circle(
        (cx, cy), INPUT_RADIUS,
        facecolor=COL_INPUT_FILL,
        edgecolor=COL_BORDER,
        linewidth=1.5,
        zorder=4,
    )
    ax.add_patch(circ)
    # Otimes symbol
    ax.text(
        cx, cy,
        r'$\otimes$',
        ha='center', va='center',
        fontsize=16,
        color='#555555',
        zorder=5,
    )
    # Label below
    ax.text(
        cx, cy - INPUT_RADIUS - 0.18,
        'New instance',
        ha='center', va='top',
        fontsize=9,
        color=COL_TEXT,
        zorder=5,
    )


def main():
    fig, ax = _style.new_ax(figsize=(7.5, 9.0))

    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.set_aspect('equal')
    ax.axis('off')

    # -----------------------------------------------------------------------
    # 1. Input node (bottom center)
    # -----------------------------------------------------------------------
    draw_input_node(ax, X_CENTER, Y_INPUT)

    # -----------------------------------------------------------------------
    # 2. Three Base Estimator boxes
    # -----------------------------------------------------------------------
    base_colors = [COL_BLUE, COL_PURPLE, COL_PINK]
    base_labels = ['Estimator 1', 'Estimator 2', 'Estimator 3']
    for i, (xb, col, lbl) in enumerate(zip(X_BASES, base_colors, base_labels)):
        draw_base_box(ax, xb, Y_BASE, col, '', lbl)

    # -----------------------------------------------------------------------
    # 3. Prediction circles
    # -----------------------------------------------------------------------
    pred_values = ['3.1', '2.7', '2.9']
    pred_colors = ['#D6EAF8', '#E8DAEF', '#FDEDEC']
    for xb, pv, pc in zip(X_BASES, pred_values, pred_colors):
        draw_circle_node(ax, xb, Y_PRED, PRED_RADIUS, pc, pv, fontsize=11)

    # -----------------------------------------------------------------------
    # 4. Blending model box (top center)
    # -----------------------------------------------------------------------
    draw_blend_box(ax, X_CENTER, Y_BLEND)

    # -----------------------------------------------------------------------
    # 5. Final output circle
    # -----------------------------------------------------------------------
    draw_circle_node(ax, X_CENTER, Y_OUTPUT, OUTPUT_RADIUS, '#D5F5E3', '3.0', fontsize=12)

    # -----------------------------------------------------------------------
    # 6. Arrows: Input -> Base estimators (fan out)
    # -----------------------------------------------------------------------
    for xb in X_BASES:
        draw_arrow(ax, X_CENTER, Y_INPUT + INPUT_RADIUS,
                   xb, Y_BASE - BASE_H / 2,
                   shrinkA=2, shrinkB=4)

    # -----------------------------------------------------------------------
    # 7. Arrows: Base estimators -> Prediction circles
    # -----------------------------------------------------------------------
    for xb in X_BASES:
        draw_arrow(ax, xb, Y_BASE + BASE_H / 2,
                   xb, Y_PRED - PRED_RADIUS,
                   shrinkA=2, shrinkB=4)

    # -----------------------------------------------------------------------
    # 8. Arrows: Prediction circles -> Blending box (converge)
    # -----------------------------------------------------------------------
    blend_top_y    = Y_BLEND - BLEND_H / 2
    blend_left_x   = X_CENTER - BLEND_W / 2 + 0.2
    blend_mid_x    = X_CENTER
    blend_right_x  = X_CENTER + BLEND_W / 2 - 0.2

    targets_x = [blend_left_x, blend_mid_x, blend_right_x]
    for xb, tx in zip(X_BASES, targets_x):
        draw_arrow(ax, xb, Y_PRED + PRED_RADIUS,
                   tx, blend_top_y,
                   shrinkA=4, shrinkB=4)

    # -----------------------------------------------------------------------
    # 9. Arrow: Blending box -> Final output
    # -----------------------------------------------------------------------
    draw_arrow(ax, X_CENTER, Y_BLEND + BLEND_H / 2,
               X_CENTER, Y_OUTPUT - OUTPUT_RADIUS,
               shrinkA=4, shrinkB=4)

    # -----------------------------------------------------------------------
    # 10. Right-side row labels
    # -----------------------------------------------------------------------
    label_x = CANVAS_W - 0.05
    ax.text(
        label_x, Y_PRED,
        'Predictions',
        ha='right', va='center',
        fontsize=10,
        color='#555555',
        style='italic',
        zorder=5,
    )
    ax.text(
        label_x, Y_BASE,
        'Predict',
        ha='right', va='center',
        fontsize=10,
        color='#555555',
        style='italic',
        zorder=5,
    )

    # -----------------------------------------------------------------------
    # 11. Top label
    # -----------------------------------------------------------------------
    ax.text(
        X_CENTER, CANVAS_H - 0.3,
        'Stacking Ensemble',
        ha='center', va='top',
        fontsize=13,
        color=COL_TEXT,
        fontweight='bold',
        zorder=5,
    )

    # Save
    _style.finalize(fig, OUT_PATH)
    print('saved:', OUT_PATH)


if __name__ == '__main__':
    main()
