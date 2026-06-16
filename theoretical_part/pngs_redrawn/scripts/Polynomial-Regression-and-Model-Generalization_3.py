"""Redraw: Polynomial-Regression-and-Model-Generalization_3

Overfit learning curve — degree-20 polynomial regression.

Horizontal axis: number of training samples (1 .. 75).
Vertical axis:   RMSE, fixed window [0, 4].

Data reproduced exactly from the source notebook
(05-Learning-Curve.ipynb Cells 2, 4, 10, 12, 15):
  np.random.seed(666), x ~ U(-3, 3),
  y = 0.5*x^2 + x + 2 + N(0, 1),
  train_test_split(random_state=10),
  PolynomialFeatures(20) + StandardScaler + LinearRegression.

Style: shared _style module (unified palette, CJK font stack,
       hidden top/right spines, light grid).
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "Polynomial-Regression-and-Model-Generalization_3.png"
)

# Palette: blue = train (C0), amber = test (C3, warm contrast).
TRAIN_COLOR = _style.PALETTE[0]  # #2563EB  blue
TEST_COLOR  = _style.PALETTE[3]  # #F59E0B  amber


def polynomial_regression(degree):
    """Degree-d polynomial regression pipeline matching the notebook."""
    return Pipeline([
        ("poly",       PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg",    LinearRegression()),
    ])


def main():
    # ---- Data generation (fixed seeds, matches notebook Cell 2 & 4) --------
    np.random.seed(666)
    x = np.random.uniform(-3.0, 3.0, size=100)
    X = x.reshape(-1, 1)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    # X_train.shape == (75, 1)

    # ---- Learning curve computation (matches notebook Cell 10) -------------
    train_scores, test_scores = [], []
    algo = polynomial_regression(20)

    for i in range(1, len(X_train) + 1):
        algo.fit(X_train[:i], y_train[:i])

        y_train_pred = algo.predict(X_train[:i])
        train_scores.append(mean_squared_error(y_train[:i], y_train_pred))

        y_test_pred = algo.predict(X_test)
        test_scores.append(mean_squared_error(y_test, y_test_pred))

    sizes      = np.arange(1, len(X_train) + 1)
    train_rmse = np.sqrt(train_scores)
    test_rmse  = np.sqrt(test_scores)

    # ---- Plot ---------------------------------------------------------------
    fig, ax = _style.new_ax(figsize=(7, 5))

    ax.plot(sizes, train_rmse, color=TRAIN_COLOR, linewidth=2.0, label="train")
    ax.plot(sizes, test_rmse,  color=TEST_COLOR,  linewidth=2.0, label="test")

    # Fix axes to [0, 4] so high early test values are clipped at the top --
    # this creates the visible spikes near y=4 that characterise overfitting.
    ax.set_xlim(0, len(X_train) + 1)
    ax.set_ylim(0, 4)

    ax.set_xlabel("训练样本数量")
    ax.set_ylabel("RMSE")
    ax.set_title("Overfit", pad=10)

    ax.legend(loc="upper right")

    _style.finalize(fig, OUT_PATH)
    print("Saved:", OUT_PATH)
    tail = slice(-12, None)
    print(f"  train tail RMSE ~ {np.median(train_rmse[tail]):.3f}")
    print(f"  test  tail RMSE ~ {np.median(test_rmse[tail]):.3f}")
    print(f"  max test RMSE   = {test_rmse.max():.2f}")


if __name__ == "__main__":
    main()
