"""Redraw of kNN_3: concept diagram for k-Nearest-Neighbors (k = 3).

Two-dimensional feature plane (tumor size on the x-axis, time on the y-axis).
Two classes of known samples are scattered:
    red  = benign    (label 0, lower-left cluster)
    blue = malignant (label 1, upper-right cluster)
A single green point in the upper-right is the new, unlabeled sample. Dashed
black lines connect it to its k = 3 nearest neighbors -- which here all happen
to be blue (malignant), so the majority vote predicts "malignant".

Style follows the shared _style module, but this is a deliberately minimal
"concept diagram": no grid, no tick numbers, just two arrowed axes (matching
kNN_1 / kNN_2 in the same series, like the original screenshot kNN_3.png).
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style
from _style import plt

# --- Known samples (reused from 01-kNN-Basics.ipynb) -----------------------
raw_data_X = [
    [3.393533211, 2.331273381],
    [3.110073483, 1.781539638],
    [1.343808831, 3.368360954],
    [3.582294042, 4.679179110],
    [2.280362439, 2.866990263],
    [7.423436942, 4.696522875],
    [5.745051997, 3.533989803],
    [9.172168622, 2.511101045],
    [7.792783481, 3.424088941],
    [7.939820817, 0.791637231],
]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

X = np.array(raw_data_X)
y = np.array(raw_data_y)

# New, unlabeled sample placed in the upper-right so its 3 nearest neighbors
# are all blue (malignant) -- echoing the text "all three points are malignant".
x_new = np.array([9.6, 4.6])

# --- Colors (red=benign=0, blue=malignant=1, green=new) --------------------
COLOR_BENIGN = "#E8505B"     # red
COLOR_MALIGNANT = "#1F77B4"  # blue
COLOR_NEW = "#2CA02C"        # green

# --- Find the k = 3 nearest neighbors (Euclidean distance, np.argsort) ------
k = 3
distances = np.sqrt(np.sum((X - x_new) ** 2, axis=1))
nearest = np.argsort(distances)[:k]

fig, ax = _style.new_ax(figsize=(7.5, 5))

# Concept diagram: drop the default grid/spines for the clean simple-line look.
ax.grid(False)
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Plot extents (give margin so arrowed axes + points sit comfortably).
x_min, x_max = 0.0, 11.5
y_min, y_max = 0.0, 6.0
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# --- Arrowed axes (origin slightly inside the data box) --------------------
ox, oy = 0.6, 0.4  # origin of the drawn axes
arrow_kw = dict(arrowstyle="-|>", color="#222222", linewidth=1.8,
                mutation_scale=18)
# x-axis arrow
ax.annotate("", xy=(x_max - 0.2, oy), xytext=(ox, oy), arrowprops=arrow_kw)
# y-axis arrow
ax.annotate("", xy=(ox, y_max - 0.15), xytext=(ox, oy), arrowprops=arrow_kw)

# Axis labels (Chinese).
ax.text(x_max - 0.2, oy - 0.28, "肿瘤大小", fontsize=14,
        ha="right", va="top", color="#222222")
ax.text(ox - 0.15, y_max - 0.1, "时间", fontsize=14,
        ha="right", va="top", color="#222222")

# --- Dashed lines from the new point to its k = 3 nearest neighbors ---------
# Drawn first (low zorder) so the big dots sit on top of the line ends.
for idx in nearest:
    ax.plot([x_new[0], X[idx, 0]], [x_new[1], X[idx, 1]],
            linestyle=(0, (1, 1.3)),  # dotted, like the original screenshot
            color="#111111", linewidth=2.0, zorder=2)

# --- Scatter the known samples ---------------------------------------------
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=420, c=COLOR_BENIGN,
           edgecolors="white", linewidths=1.5, zorder=3, label="良性")
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=420, c=COLOR_MALIGNANT,
           edgecolors="white", linewidths=1.5, zorder=3, label="恶性")

# --- New, unlabeled sample (green) -----------------------------------------
ax.scatter(x_new[0], x_new[1], s=460, c=COLOR_NEW,
           edgecolors="white", linewidths=1.5, zorder=4,
           label="新样本(待预测)")

# --- "k = 3" annotation in the empty upper-middle area ----------------------
ax.text(4.3, 5.2, "k = 3", fontsize=20, fontweight="bold",
        ha="center", va="center", color="#222222", zorder=5)

# --- Title and legend -------------------------------------------------------
ax.set_title("k近邻算法", fontsize=22, color="#E8505B", pad=16)

legend = ax.legend(loc="lower right", fontsize=11, framealpha=0.95,
                   borderpad=0.8, labelspacing=0.7,
                   markerscale=0.45)
legend.set_zorder(10)

_style.finalize(
    fig, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/kNN_3.png"
)
print("saved kNN_3.png")
