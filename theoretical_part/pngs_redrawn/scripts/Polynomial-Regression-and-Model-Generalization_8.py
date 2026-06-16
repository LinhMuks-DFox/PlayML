"""Redraw of Polynomial-Regression-and-Model-Generalization_8.

A polynomial-regression + Ridge (L2) regularization fit:

  - blue scatter: 100 one-dimensional samples drawn around the line
    y = 0.5 x + 3 with Gaussian noise (np.random.seed(42)).
  - red curve: a degree=20 polynomial model, but with strong Ridge
    regularization (alpha=100), fitted on the seed=666 train split and
    predicted over np.linspace(-3, 3, 100).

The point of the figure is that, under strong L2 regularization, the
otherwise wildly oscillating 20th-degree polynomial is "tamed" into a smooth,
nearly monotone curve -- a gentle S shape that turns up slightly at both ends
(x ~= +/-3) and stays flat in the middle. This residual mild curvature is the
hallmark of Ridge (vs. LASSO, which would flatten it further toward a line).

Data, split and hyper-parameters are reproduced exactly from the source
notebook (08-Ridge-Regreesion.ipynb): seed 42 for the data, seed 666 for the
train/test split, degree=20, alpha=100. The model is fitted on X_train only,
but every one of the 100 samples is scattered -- this matches the original
code (plot_model uses the full x, y).

Styling (palette, fonts, CJK setup, clean spines) comes from the shared
_style module; the title is rendered in Chinese to unify with the rest of the
note set, and a small annotation records the key hyper-parameters.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import matplotlib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import _style

# The unified font stack puts "PingFang HK" first, but that face is missing
# some CJK glyphs (e.g. 杂). Prefer a CJK-complete face for this figure so
# every Chinese character renders, while keeping the same overall look.
matplotlib.rcParams["font.sans-serif"] = [
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "Heiti TC",
    "Songti SC",
] + _style._FONT_STACK

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "Polynomial-Regression-and-Model-Generalization_8.png"
)

SCATTER_COLOR = _style.PALETTE[0]  # blue  #2563EB
CURVE_COLOR = _style.PALETTE[1]    # red   #EF4444

DEGREE = 20
ALPHA = 100


def ridge_regression(degree, alpha):
    """degree-th degree polynomial + standardization + Ridge(alpha) pipeline.

    Reproduces RidgeRegression() from the source notebook exactly.
    """
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("ridge_reg", Ridge(alpha=alpha)),
    ])


def main():
    # --- Synthetic data: seed 42 (must be preserved to match the screenshot).
    np.random.seed(42)
    x = np.random.uniform(-3.0, 3.0, size=100)
    X = x.reshape(-1, 1)
    y = 0.5 * x + 3 + np.random.normal(0, 1, size=100)

    # --- Train/test split: seed 666 (model is fitted on X_train only).
    np.random.seed(666)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # --- Fit the strongly-regularized degree-20 Ridge model.
    model = ridge_regression(DEGREE, ALPHA)
    model.fit(X_train, y_train)

    # --- Predict the smooth curve over the full x-range.
    X_plot = np.linspace(-3, 3, 100).reshape(100, 1)
    y_plot = model.predict(X_plot)

    # --- Draw -------------------------------------------------------------
    fig, ax = _style.new_ax(figsize=(7.2, 4.6))

    # All 100 samples (full x, y), matching the original plot_model behavior.
    ax.scatter(x, y, color=SCATTER_COLOR, s=34, alpha=0.85,
               edgecolors="white", linewidths=0.5, zorder=2,
               label="样本点 (y = 0.5x + 3 + 噪声)")

    # The tamed degree-20 Ridge fit.
    ax.plot(X_plot[:, 0], y_plot, color=CURVE_COLOR, linewidth=2.6,
            solid_capstyle="round", zorder=3,
            label=f"岭回归拟合 (degree={DEGREE}, alpha={ALPHA})")

    ax.axis([-3, 3, 0, 6])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("多项式回归 + 岭正则化 (Ridge, alpha=100)")

    ax.legend(loc="upper left", framealpha=0.9)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
