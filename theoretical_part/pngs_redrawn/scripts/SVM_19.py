import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')
import _style

import numpy as np

OUT_PATH = '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_19.png'

np.random.seed(42)

# --- SVR regression line: y = 0.6x + 1 ---
w, b = 0.6, 1.0
epsilon = 0.65

# Generate 9 scattered points along the line with Gaussian noise
x_pts = np.array([0.5, 1.2, 2.0, 2.8, 3.5, 4.2, 5.0, 5.8, 6.5])
noise = np.array([0.2, -0.3, 0.55, -0.1, 0.8, -0.75, 0.3, -0.5, 0.7])
y_pts = w * x_pts + b + noise

# --- Line range for plotting ---
x_line = np.linspace(0, 7, 300)
y_line = w * x_line + b
y_upper = y_line + epsilon
y_lower = y_line - epsilon

# --- Create figure ---
fig, ax = _style.new_ax(figsize=(6.5, 4.5))

# SVR regression line (red solid)
ax.plot(x_line, y_line, color=_style.PALETTE[1], linewidth=2.2, solid_capstyle='round')

# Epsilon tube (red dashed)
ax.plot(x_line, y_upper, color=_style.PALETTE[1], linewidth=1.4,
        linestyle='--', dashes=(5, 4))
ax.plot(x_line, y_lower, color=_style.PALETTE[1], linewidth=1.4,
        linestyle='--', dashes=(5, 4))

# Scatter points (blue filled circles)
ax.scatter(x_pts, y_pts, color=_style.PALETTE[0], s=55, zorder=5,
           edgecolors='white', linewidths=0.8)

# --- Double-headed arrow for epsilon (from center line to lower dashed line) ---
# Place the arrow at x = 1.5 for clear visibility
x_arrow = 1.5
y_mid = w * x_arrow + b
y_lo = y_mid - epsilon

ax.annotate(
    '',
    xy=(x_arrow, y_lo),
    xytext=(x_arrow, y_mid),
    arrowprops=dict(
        arrowstyle='<->',
        color='#333333',
        lw=1.5,
    )
)

# Label epsilon next to the arrow
ax.text(x_arrow + 0.12, (y_mid + y_lo) / 2,
        r'$\varepsilon$',
        va='center', ha='left',
        fontsize=13, color='#333333')

# --- Axis styling: textbook-style, hide top/right (handled by finalize),
#     show left/bottom as black, add arrow-like y-axis ---
ax.set_xlim(-0.3, 7.2)
ax.set_ylim(0.1, 5.8)

# Thicken left and bottom spines
ax.spines['left'].set_linewidth(1.6)
ax.spines['bottom'].set_linewidth(1.6)
ax.spines['left'].set_color('#222222')
ax.spines['bottom'].set_color('#222222')

# Axis labels
ax.set_xlabel(r'$x$', fontsize=12, labelpad=2)
ax.set_ylabel(r'$y$', fontsize=12, rotation=0, labelpad=10)

# Remove tick marks for a cleaner concept-diagram look
ax.tick_params(axis='both', which='both', length=0)
ax.set_xticks([])
ax.set_yticks([])

# Turn off grid for cleaner concept diagram
ax.grid(False)

_style.finalize(fig, OUT_PATH)
print(f'Saved to {OUT_PATH}')
