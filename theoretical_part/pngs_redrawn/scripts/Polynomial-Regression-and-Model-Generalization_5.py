"""Redraw of Polynomial-Regression-and-Model-Generalization_5: the
"Under fit" learning curve.

A learning curve for a degree-1 LinearRegression fitted to data that is
actually quadratic (y = 0.5*x^2 + x + 2 + noise). Because the model is too
simple, it underfits: as the number of training samples grows, BOTH the
train RMSE (blue) and the test RMSE (orange) settle at a relatively HIGH
level (~1.7 train / ~2.0 test) with only a small gap between them. That high,
flat plateau is the signature of underfitting -- a more expressive model
(e.g. degree=2) would plateau much lower.

Exact reproduction follows the chp6 notebook:
  np.random.seed(666); x = uniform(-3,3,100); y = 0.5*x**2 + x + 2 + N(0,1)
  train_test_split(random_state=10)  -> 75 train / 25 test
  LinearRegression(degree=1), incremental fit over the first i train points.
RMSE = sqrt(mean_squared_error). Axes fixed to [0, 76] x [0, 4], so the very
high test errors at small i are clipped at the top edge, just like the
original.

Style matches the unified _style.py. The original had an English title only
and no axis names; we keep the English title and add Chinese axis labels.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Polynomial-Regression-and-Model-Generalization_5.png"

TRAIN_COLOR = _style.PALETTE[0]  # blue  #2563EB
TEST_COLOR = _style.PALETTE[3]   # amber #F59E0B (closest to the original C1 orange)


def build_dataset():
    """The exact chp6 synthetic dataset (quadratic + gaussian noise)."""
    np.random.seed(666)
    x = np.random.uniform(-3.0, 3.0, size=100)
    X = x.reshape(-1, 1)
    y = 0.5 * x ** 2 + x + 2.0 + np.random.normal(0, 1, size=100)
    return train_test_split(X, y, random_state=10)


def learning_curve(estimator, X_train, X_test, y_train, y_test):
    """Incremental learning curve: fit on the first i train points, record RMSE.

    Returns (sizes, train_rmse, test_rmse). For each i the model is fit on the
    first i training samples; train RMSE is self-evaluated on those i points,
    test RMSE is always evaluated on the full test set.
    """
    sizes = range(1, len(X_train) + 1)
    train_rmse = []
    test_rmse = []
    for i in sizes:
        estimator.fit(X_train[:i], y_train[:i])
        y_train_pred = estimator.predict(X_train[:i])
        train_rmse.append(np.sqrt(mean_squared_error(y_train[:i], y_train_pred)))
        y_test_pred = estimator.predict(X_test)
        test_rmse.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    return list(sizes), train_rmse, test_rmse


def main():
    X_train, X_test, y_train, y_test = build_dataset()

    sizes, train_rmse, test_rmse = learning_curve(
        LinearRegression(), X_train, X_test, y_train, y_test
    )

    fig, ax = _style.new_ax(figsize=(6.5, 4.6))

    ax.plot(sizes, train_rmse, color=TRAIN_COLOR, linewidth=2.0, label="train")
    ax.plot(sizes, test_rmse, color=TEST_COLOR, linewidth=2.0, label="test")

    # Fixed range as in the original; high test errors at small i clip at top.
    ax.set_xlim(0, 76)
    ax.set_ylim(0, 4)
    ax.set_xticks(range(0, 71, 10))
    ax.set_yticks(np.arange(0, 4.01, 0.5))

    ax.set_title("Under fit (欠拟合)")
    ax.set_xlabel("训练样本数")
    ax.set_ylabel("误差 RMSE")

    ax.legend(loc="upper right")

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
