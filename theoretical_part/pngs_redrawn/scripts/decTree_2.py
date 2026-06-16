import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')
import _style

import matplotlib.pyplot as plt

OUT_PATH = '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/decTree_2.png'

# 4:3 figure
fig, ax = _style.new_ax(figsize=(8, 6))

# --- Data points (scatter) ---
# left-upper: blue, right-upper: blue, left-lower: red, right-lower: blue
xs = [0.12, 0.82, 0.12, 0.80]
ys = [0.80, 0.78, 0.18, 0.20]
colors = ['#2563EB', '#2563EB', '#EF4444', '#2563EB']

ax.scatter(xs, ys, c=colors, s=3000, zorder=5, edgecolors='white', linewidths=2)

# --- L-shaped decision boundary (black) ---
# Horizontal segment: separates upper from lower, stops at vertical split
ax.plot([0.05, 0.50], [0.48, 0.48], color='black', linewidth=4, solid_capstyle='round')
# Vertical segment: from split point downward, separates left-lower (red) from right-lower (blue)
ax.plot([0.50, 0.50], [0.00, 0.48], color='black', linewidth=4, solid_capstyle='round')

# --- Purple diagonal boundary ---
# The line should pass BETWEEN the red point (0.12, 0.18) and the blue clusters,
# without overlapping any dot.
# Red is bottom-left, blues are top-left, top-right, and bottom-right.
# A good separator: diagonal from upper-left area to lower-right area,
# passing between (0.12, 0.18) red and (0.80, 0.20) blue.
# Line equation: y = -x + c, find c that passes midway between them.
# Midpoint between red(0.12,0.18) and right-lower blue(0.80,0.20): x=0.46, y=0.19
# Slope -1: y = -x + c => c = 0.19 + 0.46 = 0.65
# Check: at x=0.12: y=0.53 (above red at 0.18, well below blue at 0.80) -- good
# Check: at x=0.80: y=-0.15 which is off-screen -- need to shift line slightly
# Use shallower slope to keep line in frame and visually clear
# Try slope -0.7: midpoint x=0.46, y=0.19 => y = -0.7*x + c => c = 0.19 + 0.7*0.46 = 0.512
# at x=0.12: y = 0.512 - 0.084 = 0.428 (between red at 0.18 and blue at 0.80) -- good
# at x=0.80: y = 0.512 - 0.56 = -0.048 (just below plot area)
# Adjust intercept upward so line stays visible across full plot
# Use c=0.62, slope=-0.7:
# at x=0.12: y=0.62-0.084=0.536, at x=0.80: y=0.62-0.56=0.06
# at x=0.12 red is at 0.18 and line is at 0.536 -- line is above red, good
# at x=0.80 blue is at 0.20 and line is at 0.06 -- line is below blue, good
# at x=0.12 blue is at 0.80 and line is at 0.536 -- line is below upper-left blue, good
# at x=0.82 blue is at 0.78 and line is at 0.62-0.574=0.046 -- line well below, good
# Now find endpoints: extend line to x-range [0.02, 0.92]
# x=0.02: y=0.62-0.014=0.606; x=0.92: y=0.62-0.644=-0.024
# Clip to [0,1]: find x where y=0: 0=0.62-0.7x => x=0.886
# So line goes from (0.02, 0.606) to (0.886, 0.0), clip y to [0.02, ...]
# Use (0.02, 0.606) to (0.88, 0.004) -- near bottom edge
# Shorten slightly for aesthetics: (0.05, 0.585) to (0.85, 0.025)
ax.plot([0.05, 0.85], [0.585, 0.025],
        color='#AA00FF', linewidth=4, solid_capstyle='round')

# --- Text annotations ---
ax.text(0.26, 0.36, '决策树产生的\n决策边界',
        color='black', fontsize=14, ha='center', va='center',
        fontweight='semibold')

ax.text(0.78, 0.58, '可能可以更加良好\n反应数据分类的边界',
        color='#AA00FF', fontsize=14, ha='center', va='center',
        fontweight='semibold')

# --- Remove all axes decoration ---
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

_style.finalize(fig, OUT_PATH)
print(f"Saved to {OUT_PATH}")
