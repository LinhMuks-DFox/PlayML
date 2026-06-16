"""Redraw of kNN_4: concept diagram for k-Nearest-Neighbors voting (k = 3).

Two-dimensional feature plane (tumor size on the x-axis, time on the y-axis).
Two classes of known samples are scattered:
    red  = benign    (label 0)
    blue = malignant (label 1)
A single green point near the middle-left is the new, unlabeled sample. Dashed
black lines connect the green point to its 3 nearest neighbors by Euclidean
distance -- here 2 red + 1 blue -- so the majority vote (k = 3) predicts benign.

Style follows the shared _style module, but this is a deliberately minimal
"concept diagram": no grid, no tick numbers, just two arrowed axes (like the
original screenshot kNN_4.png).
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style
from _style import plt

# --- Known samples (hand-placed fixed coordinates) -------------------------
# red = benign (0), blue = malignant (1). Layout is arranged so that the
# green point's 3 nearest neighbours are exactly 2 red + 1 blue.
raw_data_X = [
    [2.1, 3.4],   # red  - near (nearest neighbour group)
    [3.6, 2.4],   # red  - near (nearest neighbour group)
    [1.9, 1.5],   # red  - far (lower-left)
    [6.3, 2.2],   # red  - far (lower-right)
    [4.9, 4.6],   # blue - near (nearest neighbour group)
    [7.0, 6.0],   # blue - far (upper-right)
    [8.8, 5.0],   # blue - far (right)
    [7.6, 3.2],   # blue - far (lower-right)
]
raw_data_y = [0, 0, 0, 0, 1, 1, 1, 1]

X = np.array(raw_data_X)
y = np.array(raw_data_y)

# New, unlabeled sample (green), placed centre-left.
x_new = np.array([3.5, 3.6])

# --- Find the k=3 nearest neighbours by Euclidean distance -----------------
K = 3
dists = np.sqrt(((X - x_new) ** 2).sum(axis=1))
nn_idx = np.argsort(dists)[:K]
nn_labels = y[nn_idx]
n_red = int((nn_labels == 0).sum())
n_blue = int((nn_labels == 1).sum())
# Sanity check: the layout must yield 2 red + 1 blue (see notes / line 60).
assert (n_red, n_blue) == (2, 1), f"expected 2 red 1 blue, got {n_red} red {n_blue} blue"

# --- Colors (red=benign=0, blue=malignant=1, green=new) --------------------
COLOR_BENIGN = "#E8505B"     # red
COLOR_MALIGNANT = "#1F77B4"  # blue
COLOR_NEW = "#2CA02C"        # green

fig, ax = _style.new_ax(figsize=(7.5, 5))

# Concept diagram: drop the default grid/spines for the clean simple-line look.
ax.grid(False)
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Plot extents (give margin so arrowed axes + points sit comfortably).
x_min, x_max = 0.0, 10.5
y_min, y_max = 0.0, 7.2
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# --- Arrowed axes (origin slightly inside the data box) --------------------
ox, oy = 0.6, 0.45  # origin of the drawn axes
arrow_kw = dict(arrowstyle="-|>", color="#222222", linewidth=1.8,
                mutation_scale=18)
# x-axis arrow
ax.annotate("", xy=(x_max - 0.2, oy), xytext=(ox, oy), arrowprops=arrow_kw)
# y-axis arrow
ax.annotate("", xy=(ox, y_max - 0.15), xytext=(ox, oy), arrowprops=arrow_kw)

# Axis labels (Chinese).
ax.text(x_max - 0.2, oy - 0.3, "肿瘤大小", fontsize=14,
        ha="right", va="top", color="#222222")
ax.text(ox - 0.15, y_max - 0.1, "时间", fontsize=14,
        ha="right", va="top", color="#222222")

# --- Dashed lines from the new point to its k nearest neighbours -----------
# Drawn first (low zorder) so the big dots sit on top.
for i in nn_idx:
    ax.plot([x_new[0], X[i, 0]], [x_new[1], X[i, 1]],
            linestyle=(0, (2, 2)), color="#222222", linewidth=1.6, zorder=2)

# --- Scatter the known samples ---------------------------------------------
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=480, c=COLOR_BENIGN,
           edgecolors="white", linewidths=1.5, zorder=3, label="良性")
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=480, c=COLOR_MALIGNANT,
           edgecolors="white", linewidths=1.5, zorder=3, label="恶性")

# --- New, unlabeled sample (green) -----------------------------------------
ax.scatter(x_new[0], x_new[1], s=520, c=COLOR_NEW,
           edgecolors="white", linewidths=1.5, zorder=4,
           label="新样本(待预测)")

# --- "k = 3" annotation above the green point ------------------------------
ax.text(x_new[0], x_new[1] + 1.15, "k = 3", fontsize=17, fontweight="bold",
        ha="center", va="bottom", color="#222222", zorder=5)

# --- Title and legend -------------------------------------------------------
ax.set_title("k近邻算法", fontsize=22, color="#E8505B", pad=16)

legend = ax.legend(loc="lower right", fontsize=11, framealpha=0.95,
                   borderpad=0.8, labelspacing=0.7,
                   markerscale=0.42)
legend.set_zorder(10)

_style.finalize(
    fig, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/kNN_4.png"
)
print(f"saved kNN_4.png  (nearest {K}: {n_red} red, {n_blue} blue -> benign)")
