import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')
import _style

import numpy as np
import matplotlib.pyplot as plt

# --- Landmark positions ---
l1 = -1.0
l2 = 1.0

# --- Explicit point layout: red-red-red - l1(△) - blue-blue-blue - l2(△) - red-red-red ---
# Red (negative class, class 0): left side (x < l1) and right side (x > l2)
x_class0 = np.array([-4.0, -3.0, -2.0, 2.0, 3.0, 4.0])

# Blue (positive class, class 1): between l1 and l2 (exclusive of landmark positions)
x_class1 = np.array([-0.5, 0.0, 0.5])

# --- Figure ---
fig, ax = _style.new_ax(figsize=(8, 2.2))

# Hide all default spines and ticks
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-0.6, 0.9)
ax.grid(False)

# Draw number line: a horizontal line with an arrow at the right end
line_start = -5.0
line_end = 5.2
ax.annotate(
    '',
    xy=(line_end, 0),
    xytext=(line_start, 0),
    arrowprops=dict(
        arrowstyle='->', color='#333333', lw=1.8,
        mutation_scale=14
    )
)

# Tick marks at integer positions
for tx in np.arange(-4, 5, 1):
    ax.plot([tx, tx], [-0.07, 0.07], color='#555555', lw=1.0, zorder=3)
    ax.text(tx, -0.20, str(int(tx)), ha='center', va='top',
            fontsize=9, color='#555555')

# Plot class 0 (negative / red circles) at y=0
ax.scatter(
    x_class0, np.zeros_like(x_class0),
    s=120, color='#EF4444', marker='o', zorder=5,
    label='class 0'
)

# Plot class 1 (positive / blue circles) at y=0
ax.scatter(
    x_class1, np.zeros_like(x_class1),
    s=120, color='#2563EB', marker='o', zorder=5,
    label='class 1'
)

# Plot landmarks (purple triangles), slightly larger
ax.scatter(
    [l1, l2], [0, 0],
    s=180, color='#8B5CF6', marker='^', zorder=6,
    label='landmark'
)

# Annotate l1 and l2 labels above the triangles
label_y = 0.38
ax.text(
    l1, label_y, r'$l_1$',
    fontsize=15, ha='center', va='bottom',
    color='#8B5CF6',
    style='italic'
)
ax.text(
    l2, label_y, r'$l_2$',
    fontsize=15, ha='center', va='bottom',
    color='#8B5CF6',
    style='italic'
)

# Optional legend
ax.legend(
    loc='upper right',
    bbox_to_anchor=(1.0, 1.05),
    fontsize=9,
    markerscale=0.85
)

# finalize: override the default spine removal to keep the clean look
# (spines already hidden manually above, so finalize's spine pass is harmless)
_style.finalize(fig, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_17.png')
