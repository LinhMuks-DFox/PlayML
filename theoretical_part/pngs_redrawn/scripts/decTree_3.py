import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')
import _style

import numpy as np

# --- Generate synthetic linearly separable data ---
rng = np.random.default_rng(42)

# Class 0 (blue): left side, x in [0.1, 0.45], y in [0.1, 0.9]
n0 = 8
x0 = rng.uniform(0.1, 0.45, n0)
y0 = rng.uniform(0.1, 0.9, n0)

# Class 1 (red): right side, x in [0.55, 0.95], y in [0.1, 0.9]
n1 = 9
x1 = rng.uniform(0.55, 0.95, n1)
y1 = rng.uniform(0.1, 0.9, n1)

# Decision boundary at x = 0.5
boundary_x = 0.5

# --- Plot ---
fig, ax = _style.new_ax(figsize=(6, 4.5))

# Scatter: class 0 (blue)
ax.scatter(x0, y0, color='#4C9BE8', s=180, marker='o',
           zorder=3, label='类别 0', edgecolors='white', linewidths=0.8)

# Scatter: class 1 (red)
ax.scatter(x1, y1, color='#D95A3B', s=180, marker='o',
           zorder=3, label='类别 1', edgecolors='white', linewidths=0.8)

# Vertical decision boundary
ax.axvline(x=boundary_x, color='black', linewidth=3, zorder=2)

# Clean up axes: remove tick labels, keep spine lines
ax.set_xticks([])
ax.set_yticks([])

# Turn off grid for this clean concept diagram
ax.grid(False)

# Set axis limits with a small margin
ax.set_xlim(0.0, 1.05)
ax.set_ylim(0.0, 1.0)

# Keep left and bottom spines, remove top and right (finalize handles top/right)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# Optional simple legend
ax.legend(loc='upper center', ncol=2, markerscale=0.9, handletextpad=0.4,
          columnspacing=1.0)

_style.finalize(fig, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/decTree_3.png')
