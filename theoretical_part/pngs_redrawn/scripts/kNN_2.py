"""Redraw of kNN_2: kNN concept diagram with query point.

Two-dimensional feature plane (tumor size x-axis, time y-axis).
Training samples (10 points, exactly from 01-kNN-Basics.ipynb cell-5):
    red  = benign    (label 0, lower-left cluster)
    blue = malignant (label 1, right cluster)
One green query point sits in an empty area within the blue cluster --
visually distinct from all training points, with its k=3 nearest
neighbors all being blue (malignant), consistent with notebook notes
("these three points are all malignant").

No legend text, no tick numbers, no grid -- clean concept-diagram style
with arrow-tipped axes, matching the original kNN_2.png screenshot.

QA fixes applied:
- Green point is NOT overlapping any training point (min dist ~0.44).
- Green point k=3 nearest neighbors are all blue (malignant).
- Red (benign) points stay in lower-left area; blue in right area.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style
from _style import plt

# ---------------------------------------------------------------------------
# Data: exactly from 01-kNN-Basics.ipynb cell-5
# ---------------------------------------------------------------------------
raw_data_X = [
    [3.393533211, 2.331273381],   # benign (0) -- lower-left
    [3.110073483, 1.781539638],   # benign (0) -- lower-left
    [1.343808831, 3.368360954],   # benign (0) -- left
    [3.582294042, 4.679179110],   # benign (0) -- left-upper
    [2.280362439, 2.866990263],   # benign (0) -- lower-left
    [7.423436942, 4.696522875],   # malignant (1) -- upper-right
    [5.745051997, 3.533989803],   # malignant (1) -- middle-right
    [9.172168622, 2.511101045],   # malignant (1) -- far right
    [7.792783481, 3.424088941],   # malignant (1) -- right
    [7.939820817, 0.791637231],   # malignant (1) -- lower-right
]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

X = np.array(raw_data_X)
y = np.array(raw_data_y)

# Green query point:
# - Clearly separated from all training points (min distance ~1.12).
# - k=3 nearest neighbors are all blue (malignant): indices 5, 8, 7.
# - Sits in the blue cluster area (upper-right) as a clearly visible newcomer.
x_new = np.array([8.5, 5.0])

# ---------------------------------------------------------------------------
# Colors: red = benign (0), blue = malignant (1), green = new query point
# ---------------------------------------------------------------------------
COLOR_BENIGN = "#EF4444"     # red  (palette[1])
COLOR_MALIGNANT = "#2563EB"  # blue (palette[0])
COLOR_NEW = "#10B981"        # green (palette[2])

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = _style.new_ax(figsize=(7.5, 5))

# Concept diagram: no grid, no spines, no tick numbers.
ax.grid(False)
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Plot limits with enough margin for arrows and labels.
x_min, x_max = 0.0, 11.0
y_min, y_max = 0.0, 6.5
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# ---------------------------------------------------------------------------
# Arrow-tipped axes  (origin at (0.4, 0.3) for visual breathing room)
# ---------------------------------------------------------------------------
ox, oy = 0.4, 0.3
arrow_kw = dict(
    arrowstyle="-|>",
    color="#222222",
    linewidth=1.8,
    mutation_scale=20,
)
ax.annotate("", xy=(x_max - 0.15, oy), xytext=(ox, oy), arrowprops=arrow_kw)
ax.annotate("", xy=(ox, y_max - 0.1), xytext=(ox, oy), arrowprops=arrow_kw)

# Axis labels near the arrowheads.
ax.text(x_max - 0.15, oy - 0.27, "肿瘤大小",
        fontsize=14, ha="right", va="top", color="#222222")
ax.text(ox - 0.12, y_max - 0.05, "时间",
        fontsize=14, ha="right", va="top", color="#222222")

# ---------------------------------------------------------------------------
# Training samples
# Red (benign) points: lower-left cluster (x < 5, generally y < 5)
# Blue (malignant) points: right cluster (x > 5)
# ---------------------------------------------------------------------------
ax.scatter(X[y == 0, 0], X[y == 0, 1],
           s=380, c=COLOR_BENIGN,
           edgecolors="white", linewidths=1.5, zorder=3)
ax.scatter(X[y == 1, 0], X[y == 1, 1],
           s=380, c=COLOR_MALIGNANT,
           edgecolors="white", linewidths=1.5, zorder=3)

# ---------------------------------------------------------------------------
# New query point (green) -- in blue cluster area, clearly distinct
# from all training points, k=3 nearest are all blue (indices 5, 8, 6)
# ---------------------------------------------------------------------------
ax.scatter(x_new[0], x_new[1],
           s=420, c=COLOR_NEW,
           edgecolors="white", linewidths=1.5, zorder=4)

# ---------------------------------------------------------------------------
# Title: red, bold, Chinese
# ---------------------------------------------------------------------------
ax.set_title("k近邻算法", fontsize=22, color=COLOR_BENIGN, fontweight="bold", pad=16)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
_style.finalize(
    fig,
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/kNN_2.png",
)
print("saved kNN_2.png")
