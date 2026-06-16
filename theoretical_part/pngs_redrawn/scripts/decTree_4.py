import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')
import _style

import numpy as np

# --- Generate the same synthetic data as decTree_3, then rotate ~45 degrees ---
rng = np.random.default_rng(42)

# Class 0 (blue): left side in original space
n0 = 8
x0 = rng.uniform(0.1, 0.45, n0)
y0 = rng.uniform(0.1, 0.9, n0)

# Class 1 (red): right side in original space
n1 = 9
x1 = rng.uniform(0.55, 0.95, n1)
y1 = rng.uniform(0.1, 0.9, n1)

# Stack and center both classes for rotation
X0 = np.column_stack([x0, y0])
X1 = np.column_stack([x1, y1])
X_all = np.vstack([X0, X1])

# Center around (0.5, 0.5) before rotation
center = np.array([0.5, 0.5])
X0_c = X0 - center
X1_c = X1 - center

# Rotation matrix for ~45 degrees
angle = np.deg2rad(45)
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle),  np.cos(angle)]])

X0_rot = X0_c @ R.T + center
X1_rot = X1_c @ R.T + center

# Rescale so that data fits nicely in [0.05, 0.95] range
all_rot = np.vstack([X0_rot, X1_rot])
mn = all_rot.min(axis=0)
mx = all_rot.max(axis=0)
scale = mx - mn
X0_rot = (X0_rot - mn) / scale * 0.88 + 0.05
X1_rot = (X1_rot - mn) / scale * 0.88 + 0.05

# --- Plot ---
fig, ax = _style.new_ax(figsize=(6, 4.5))

# Scatter: class 0 (blue) - now in lower-left after rotation
ax.scatter(X0_rot[:, 0], X0_rot[:, 1], color='#4C9BE8', s=180, marker='o',
           zorder=3, edgecolors='white', linewidths=0.8)

# Scatter: class 1 (red/tomato) - now in upper-right after rotation
ax.scatter(X1_rot[:, 0], X1_rot[:, 1], color='#D95A3B', s=180, marker='o',
           zorder=3, edgecolors='white', linewidths=0.8)

# --- Good decision boundary: a diagonal line (magenta/pink) ---
# The natural boundary between the two diagonal clusters is a line
# perpendicular to the separation direction (roughly y = -x + 1 after rescaling)
# We draw from upper-left to lower-right to separate lower-left blue from upper-right red
diag_x = np.array([0.0, 1.05])
# slope ~ -1, intercept chosen so line passes between the two clusters
# The clusters are separated along the diagonal, boundary passes through center ~(0.5, 0.5)
diag_y = -1.0 * diag_x + 1.08
ax.plot(diag_x, diag_y, color='magenta', linewidth=2.5, zorder=2,
        linestyle='-', label='良好的决策边界')

# --- Decision tree boundary: staircase (axis-aligned segments) ---
# Hand-crafted staircase that roughly follows the diagonal boundary
# Going from upper-left to lower-right in axis-aligned steps
stair_x = [0.08, 0.30, 0.30, 0.52, 0.52, 0.72, 0.72, 1.00]
stair_y = [0.75, 0.75, 0.53, 0.53, 0.33, 0.33, 0.15, 0.15]

ax.plot(stair_x, stair_y, color='black', linewidth=2.5, zorder=2,
        linestyle='-', label='决策树产生的决策边界')

# --- Text annotations ---
ax.text(0.10, 0.96, '良好的决策边界',
        color='magenta', fontsize=11, va='top', ha='left',
        transform=ax.transAxes)

ax.text(0.02, 0.50, '决策树产生的\n决策边界',
        color='black', fontsize=11, va='center', ha='left',
        transform=ax.transAxes)

# --- Clean up axes ---
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)

ax.set_xlim(0.0, 1.05)
ax.set_ylim(0.0, 1.05)

ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

_style.finalize(fig, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/decTree_4.png')
