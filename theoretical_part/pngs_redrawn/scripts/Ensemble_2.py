import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')
import _style

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# --- Reproducible synthetic data --------------------------------------------
np.random.seed(42)
n = 100
X = np.sort(np.random.uniform(-0.5, 0.5, n))

# Piecewise-constant-ish nonlinear function + Gaussian noise
def true_func(x):
    y = np.zeros_like(x)
    y += 0.7 * (x < -0.3)
    y += 0.5 * ((x >= -0.3) & (x < -0.1))
    y += 0.2 * ((x >= -0.1) & (x <  0.1))
    y += 0.5 * ((x >=  0.1) & (x <  0.3))
    y += 0.7 * (x >= 0.3)
    return y

y_clean = true_func(X)
noise = np.random.normal(0, 0.05, n)
y = y_clean + noise

X2d = X.reshape(-1, 1)

# --- Fit three trees iteratively on residuals --------------------------------
tree1 = DecisionTreeRegressor(max_depth=2, random_state=0)
tree1.fit(X2d, y)

res1 = y - tree1.predict(X2d)

tree2 = DecisionTreeRegressor(max_depth=2, random_state=1)
tree2.fit(X2d, res1)

res2 = res1 - tree2.predict(X2d)

tree3 = DecisionTreeRegressor(max_depth=2, random_state=2)
tree3.fit(X2d, res2)

# --- Dense x-grid for smooth step curves ------------------------------------
x_dense = np.linspace(-0.5, 0.5, 1000).reshape(-1, 1)

h1 = tree1.predict(x_dense)
h2 = tree2.predict(x_dense)
h3 = tree3.predict(x_dense)

# Ensemble cumulative predictions on dense grid
ens1 = h1
ens2 = h1 + h2
ens3 = h1 + h2 + h3

# --- Layout -----------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(10, 9))

# Palette from _style
BLUE    = _style.PALETTE[0]   # #2563EB
RED     = _style.PALETTE[1]   # #EF4444
GREEN   = _style.PALETTE[2]   # #10B981
GRAY    = _style.PALETTE[7]   # #64748B

scatter_kw = dict(s=18, alpha=0.55, zorder=3)
curve_kw   = dict(linewidth=2.0, zorder=4)

# ============================================================
# Row 0 — M1: original training data / first tree / ensemble1
# ============================================================
ax = axes[0, 0]
ax.scatter(X, y, color=BLUE, **scatter_kw, label='Training set')
ax.plot(x_dense, h1, color=GREEN, **curve_kw, label=r'$h_1(x_1)$')
ax.set_ylabel(r'$y$', fontsize=11)
ax.set_xlabel(r'$x_1$', fontsize=10)
ax.legend(fontsize=8.5, loc='upper left')

ax = axes[0, 1]
ax.scatter(X, y, color=BLUE, **scatter_kw, label='Training set')
ax.plot(x_dense, ens1, color=RED, **curve_kw, label=r'$h(x_1)=h_1(x_1)$')
ax.set_ylabel(r'$y$', fontsize=11)
ax.set_xlabel(r'$x_1$', fontsize=10)
ax.legend(fontsize=8.5, loc='upper left')

# ============================================================
# Row 1 — M2: residuals after h1 / second tree / ensemble2
# ============================================================
ax = axes[1, 0]
ax.scatter(X, res1, color=GRAY, **scatter_kw, label='Residuals')
ax.plot(x_dense, h2, color=GREEN, **curve_kw, label=r'$h_2(x_1)$')
ax.set_ylabel(r'$y - h_1(x_1)$', fontsize=10)
ax.set_xlabel(r'$x_1$', fontsize=10)
ax.legend(fontsize=8.5, loc='upper left')

ax = axes[1, 1]
ax.scatter(X, y, color=BLUE, **scatter_kw, label='Training set')
ax.plot(x_dense, ens2, color=RED, **curve_kw, label=r'$h(x_1)=h_1+h_2$')
ax.set_ylabel(r'$y$', fontsize=11)
ax.set_xlabel(r'$x_1$', fontsize=10)
ax.legend(fontsize=8.5, loc='upper left')

# ============================================================
# Row 2 — M3: residuals after h1+h2 / third tree / ensemble3
# ============================================================
ax = axes[2, 0]
ax.scatter(X, res2, color=GRAY, **scatter_kw, label='Residuals')
ax.plot(x_dense, h3, color=GREEN, **curve_kw, label=r'$h_3(x_1)$')
ax.set_ylabel(r'$y - h_1(x_1) - h_2(x_1)$', fontsize=9)
ax.set_xlabel(r'$x_1$', fontsize=10)
ax.legend(fontsize=8.5, loc='upper left')

ax = axes[2, 1]
ax.scatter(X, y, color=BLUE, **scatter_kw, label='Training set')
ax.plot(x_dense, ens3, color=RED, **curve_kw, label=r'$h(x_1)=h_1+h_2+h_3$')
ax.set_ylabel(r'$y$', fontsize=11)
ax.set_xlabel(r'$x_1$', fontsize=10)
ax.legend(fontsize=8.5, loc='upper left')

# ============================================================
# Column titles
# ============================================================
axes[0, 0].set_title('Residuals and tree predictions', fontsize=12, pad=6)
axes[0, 1].set_title('Ensemble predictions', fontsize=12, pad=6)

# ============================================================
# Row labels M1 / M2 / M3 on the left side
# ============================================================
row_labels = ['M1', 'M2', 'M3']
for row_idx, label in enumerate(row_labels):
    fig.text(
        0.01,
        axes[row_idx, 0].get_position().y0 + axes[row_idx, 0].get_position().height / 2,
        label,
        fontsize=13,
        fontweight='bold',
        va='center',
        ha='left',
        color='#333333',
    )

# ============================================================
# Spine cleanup + spacing
# ============================================================
for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.subplots_adjust(hspace=0.55, wspace=0.38, left=0.11, right=0.97,
                    top=0.94, bottom=0.07)

# --- Save -------------------------------------------------------------------
fig.savefig(
    '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Ensemble_2.png',
    dpi=200,
    bbox_inches='tight',
)
plt.close(fig)
