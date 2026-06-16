"""Redraw of Polynomial-Regression-and-Model-Generalization_7.

Faithfully reproduces the original notebook (09-LASSO.ipynb) figure:
- Title: 'LASSO, When alpha=0.1, Polynomial Regression'
- Blue scatter of 100 synthetic points (seed 42, train_test_split seed 666)
- Red LASSO fitted curve (degree=20 polynomial pipeline, alpha=0.1)
- Axis range: x in [-3, 3], y in [0, 6]
- No axis labels, no legend, no annotations

Data exactly matches the source notebook:
    np.random.seed(42); x = U(-3, 3, 100); y = 0.5*x + 3 + N(0,1)
    train_test_split with np.random.seed(666); fit on TRAIN split only.
    Curve predicted on np.linspace(-3, 3, 100).

Style follows the shared _style module.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import warnings

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import _style
from _style import plt

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "Polynomial-Regression-and-Model-Generalization_7.png"
)

# --- Colors from unified palette -------------------------------------------
POINT_COLOR = _style.PALETTE[0]  # blue  #2563EB
CURVE_COLOR = _style.PALETTE[1]  # red   #EF4444


def lasso_regression(degree, alpha):
    """Pipeline: PolynomialFeatures(degree) -> StandardScaler -> Lasso."""
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lasso_reg", Lasso(alpha=alpha, max_iter=100000)),
    ])


def main():
    # --- Synthetic data: linear trend + Gaussian noise (seed 42) ------------
    np.random.seed(42)
    x = np.random.uniform(-3.0, 3.0, size=100)
    X = x.reshape(-1, 1)
    y = 0.5 * x + 3.0 + np.random.normal(0, 1, size=100)

    # --- Train/test split (seed 666) -> fit on TRAIN only -------------------
    np.random.seed(666)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # --- Fit degree-20 polynomial + LASSO(alpha=0.1) ------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = lasso_regression(degree=20, alpha=0.1)
        model.fit(X_train, y_train)

        x_plot = np.linspace(-3.0, 3.0, 100).reshape(-1, 1)
        y_plot = model.predict(x_plot)

    fig, ax = _style.new_ax(figsize=(7.2, 5.0))

    # --- All samples scatter (blue) -----------------------------------------
    ax.scatter(
        x, y, s=46, c=POINT_COLOR, marker="o",
        edgecolors="none", alpha=0.9, zorder=3,
    )

    # --- LASSO model curve (red): nearly a straight line --------------------
    ax.plot(
        x_plot.ravel(), y_plot, color=CURVE_COLOR, linewidth=2.2,
        zorder=4,
    )

    # --- Axes ranges & ticks (match plt.axis([-3, 3, 0, 6])) ---------------
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(0.0, 6.0)
    ax.set_xticks(range(-3, 4))
    ax.set_yticks(range(0, 7))

    # --- No axis labels (original has none) ---------------------------------
    ax.set_xlabel("")
    ax.set_ylabel("")

    # --- Title exactly as original ------------------------------------------
    ax.set_title("LASSO, When alpha=0.1, Polynomial Regression", pad=10)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
