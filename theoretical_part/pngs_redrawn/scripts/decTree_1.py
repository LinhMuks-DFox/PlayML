import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')
import _style

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# --- Data -------------------------------------------------------------------
iris = load_iris()
X = iris.data[:, 2:]   # petal length, petal width
y = iris.target

# --- Model ------------------------------------------------------------------
clf = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=42)
clf.fit(X, y)

# --- Decision-boundary mesh -------------------------------------------------
axis = [0.5, 7.5, 0.0, 3.0]
x0 = np.linspace(axis[0], axis[1], 500)
x1 = np.linspace(axis[2], axis[3], 500)
x0_grid, x1_grid = np.meshgrid(x0, x1)
X_mesh = np.c_[x0_grid.ravel(), x1_grid.ravel()]
y_pred = clf.predict(X_mesh).reshape(x0_grid.shape)

# --- Colors -----------------------------------------------------------------
# Background regions: setosa=pink, versicolor=yellow, virginica=light-blue
bg_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
# Scatter colors aligned with classes 0/1/2
scatter_colors = ['#C62828', '#F57F17', '#1565C0']  # dark red / dark amber / dark blue
class_names = ['Setosa', 'Versicolor', 'Virginica']

# --- Figure -----------------------------------------------------------------
fig, ax = _style.new_ax(figsize=(7, 5))

# Background fill
ax.contourf(x0_grid, x1_grid, y_pred, alpha=0.55, cmap=bg_cmap)

# Scatter: original samples
for cls_idx, (color, name) in enumerate(zip(scatter_colors, class_names)):
    mask = y == cls_idx
    ax.scatter(X[mask, 0], X[mask, 1],
               color=color, edgecolors='white', linewidths=0.6,
               s=55, label=name, zorder=3)

# Decision boundary lines (dashed) for visual clarity
# First split: petal length ~ 2.45
split_x = 2.45
ax.axvline(x=split_x, color='#333333', linestyle='--', linewidth=1.4, zorder=4)

# Second split: petal width ~ 1.75 (only in x > split_x region)
split_y = 1.75
ax.plot([split_x, axis[1]], [split_y, split_y],
        color='#333333', linestyle='--', linewidth=1.4, zorder=4)

# Condition annotations
ax.text(split_x + 0.08, 2.85, 'petal length < 2.45?',
        fontsize=9, color='#333333', ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
ax.text(axis[1] - 0.1, split_y + 0.07, 'petal width < 1.75?',
        fontsize=9, color='#333333', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

# Region labels
ax.text(1.3, 1.5, 'Setosa',
        fontsize=10, color='#B71C1C', ha='center', va='center', fontweight='bold')
ax.text(4.5, 0.6, 'Versicolor',
        fontsize=10, color='#E65100', ha='center', va='center', fontweight='bold')
ax.text(5.8, 2.3, 'Virginica',
        fontsize=10, color='#0D47A1', ha='center', va='center', fontweight='bold')

# Axes labels and limits
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.set_title('Decision Tree (max_depth=2, entropy) — Iris Dataset')
ax.set_xlim(axis[0], axis[1])
ax.set_ylim(axis[2], axis[3])
ax.legend(loc='upper left', markerscale=1.2)

# --- Save -------------------------------------------------------------------
_style.finalize(fig, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/decTree_1.png')
