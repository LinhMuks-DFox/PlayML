import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')
import _style

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Figure & axes setup  (concept diagram — no coordinate axes needed)
# ---------------------------------------------------------------------------
fig, ax = _style.new_ax(figsize=(8, 7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.grid(False)

# ---------------------------------------------------------------------------
# Color scheme (soft pastels, distinct per layer)
# ---------------------------------------------------------------------------
C_INPUT  = '#D5D8DC'   # light gray  – input node
C_L1     = '#AED6F1'   # steel blue  – Layer 1
C_L2_0   = '#F1948A'   # salmon/rose – Layer 2 left
C_L2_1   = '#85C1E9'   # sky blue    – Layer 2 center
C_L2_2   = '#FAD7A0'   # peach       – Layer 2 right
C_L3     = '#A9DFBF'   # mint green  – Layer 3
C_OUT    = '#F9E79F'   # pale yellow – output bubble
EDGE_CLR = '#4A4A4A'
TEXT_CLR = '#1A1A1A'
GEAR     = 'M'     # placeholder glyph (model)

# ---------------------------------------------------------------------------
# Node layout (x, y in axes coordinates)
# ---------------------------------------------------------------------------
# y positions for each layer
Y_INPUT = 0.06
Y_L1    = 0.23
Y_L2    = 0.53
Y_L3    = 0.76
Y_OUT   = 0.92

# x positions
X_L1    = [0.22, 0.50, 0.78]   # 3 nodes in Layer 1
X_L2    = [0.22, 0.50, 0.78]   # 3 nodes in Layer 2
X_L3    = [0.50]               # 1 node in Layer 3
X_INPUT = [0.50]               # 1 input node

NODE_W  = 0.12   # bbox half-width  (used as shrink reference)
NODE_H  = 0.065  # bbox half-height

# Layer 1 annotations (left→right)
L1_VALS = ['3.1', '2.7', '2.9']
L2_VALS = ['2.9', '2.9', '3.0']
L2_COLS = [C_L2_0, C_L2_1, C_L2_2]

# ---------------------------------------------------------------------------
# Helper: draw a fancy rounded-rectangle node
# ---------------------------------------------------------------------------
def draw_node(ax, cx, cy, w, h, color, label, val=None, fontsize=10):
    """Draw a rounded-rectangle node centred at (cx, cy)."""
    box = FancyBboxPatch(
        (cx - w, cy - h),
        2 * w, 2 * h,
        boxstyle='round,pad=0.015',
        linewidth=1.4,
        edgecolor=EDGE_CLR,
        facecolor=color,
        zorder=3,
        transform=ax.transData,
        clip_on=False,
    )
    ax.add_patch(box)

    # Model icon: small filled circle + label letter
    ax.plot(cx, cy + 0.012, 'o', ms=7, color='#6C7A89',
            zorder=4, transform=ax.transData)

    # Prediction value below gear (smaller font)
    if val is not None:
        ax.text(cx, cy - 0.030, val,
                ha='center', va='center',
                fontsize=8.5, color=TEXT_CLR, fontweight='bold',
                zorder=4, transform=ax.transData)


# ---------------------------------------------------------------------------
# Helper: draw an arrow between two nodes
# ---------------------------------------------------------------------------
ARROW_KW = dict(
    arrowstyle='-|>',
    color='#5D6D7E',
    lw=1.0,
    mutation_scale=10,
    alpha=0.65,
    connectionstyle='arc3,rad=0.0',
    zorder=2,
)

def draw_arrow(ax, x0, y0, x1, y1, shrinkA=14, shrinkB=14):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        shrinkA=shrinkA, shrinkB=shrinkB,
        transform=ax.transData,
        **ARROW_KW,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Draw INPUT node (single circle with 'X' mark)
# ---------------------------------------------------------------------------
inp_x = X_INPUT[0]
inp_y = Y_INPUT

input_circle = plt.Circle((inp_x, inp_y), 0.045,
                            color=C_INPUT, ec=EDGE_CLR, lw=1.4, zorder=3)
ax.add_patch(input_circle)
ax.text(inp_x, inp_y, 'X', ha='center', va='center',
        fontsize=13, color='#666666', fontweight='bold', zorder=4)
ax.text(inp_x, inp_y - 0.07, '输入数据',
        ha='center', va='top', fontsize=8.5,
        color=TEXT_CLR, fontstyle='italic', zorder=4)

# ---------------------------------------------------------------------------
# Draw LAYER 1 nodes
# ---------------------------------------------------------------------------
for i, (x, val) in enumerate(zip(X_L1, L1_VALS)):
    draw_node(ax, x, Y_L1, NODE_W, NODE_H, C_L1, '', val=val)

# ---------------------------------------------------------------------------
# Draw LAYER 2 nodes
# ---------------------------------------------------------------------------
for i, (x, val, col) in enumerate(zip(X_L2, L2_VALS, L2_COLS)):
    draw_node(ax, x, Y_L2, NODE_W, NODE_H, col, '', val=val)

# ---------------------------------------------------------------------------
# Draw LAYER 3 node
# ---------------------------------------------------------------------------
draw_node(ax, X_L3[0], Y_L3, NODE_W, NODE_H, C_L3, '', val=None)

# ---------------------------------------------------------------------------
# Draw OUTPUT bubble
# ---------------------------------------------------------------------------
out_x = X_L3[0]
out_y = Y_OUT
out_circle = plt.Circle((out_x, out_y), 0.050,
                          color=C_OUT, ec=EDGE_CLR, lw=1.4, zorder=3)
ax.add_patch(out_circle)
ax.text(out_x, out_y, '2.9',
        ha='center', va='center',
        fontsize=11, color=TEXT_CLR, fontweight='bold', zorder=4)

# ---------------------------------------------------------------------------
# Arrows: Input → Layer 1  (3 arrows)
# ---------------------------------------------------------------------------
for x1 in X_L1:
    draw_arrow(ax, inp_x, inp_y, x1, Y_L1 - NODE_H, shrinkA=12, shrinkB=4)

# ---------------------------------------------------------------------------
# Arrows: Layer 1 → Layer 2  (9 arrows, full connection)
# ---------------------------------------------------------------------------
for x0 in X_L1:
    for x1 in X_L2:
        draw_arrow(ax, x0, Y_L1 + NODE_H, x1, Y_L2 - NODE_H, shrinkA=4, shrinkB=4)

# ---------------------------------------------------------------------------
# Arrows: Layer 2 → Layer 3  (3 arrows)
# ---------------------------------------------------------------------------
for x0 in X_L2:
    draw_arrow(ax, x0, Y_L2 + NODE_H, X_L3[0], Y_L3 - NODE_H, shrinkA=4, shrinkB=4)

# ---------------------------------------------------------------------------
# Arrow: Layer 3 → Output bubble
# ---------------------------------------------------------------------------
draw_arrow(ax, X_L3[0], Y_L3 + NODE_H, out_x, out_y - 0.05, shrinkA=4, shrinkB=6)

# ---------------------------------------------------------------------------
# Layer label annotations (left side)
# ---------------------------------------------------------------------------
label_x = 0.04
LABEL_KW = dict(ha='right', va='center', fontsize=11, fontweight='bold',
                color='#2C3E50', zorder=5)
ax.text(label_x, Y_L1,  'Layer 1', **LABEL_KW)
ax.text(label_x, Y_L2,  'Layer 2', **LABEL_KW)
ax.text(label_x, Y_L3,  'Layer 3', **LABEL_KW)

# Decorative horizontal rule next to each label
for y_lbl in [Y_L1, Y_L2, Y_L3]:
    ax.plot([label_x + 0.005, 0.08], [y_lbl, y_lbl],
            lw=1.0, color='#BDC3C7', zorder=1, transform=ax.transData)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.set_title('Stacking 集成：多层 Blender 结构', fontsize=13,
             fontweight='bold', pad=14, color='#1A1A1A')

# ---------------------------------------------------------------------------
# Finalize & save
# ---------------------------------------------------------------------------
_style.finalize(fig, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Ensemble_5.png')
