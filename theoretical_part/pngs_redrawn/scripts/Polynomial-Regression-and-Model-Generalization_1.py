"""Redraw of Polynomial-Regression-and-Model-Generalization_1.

Concept: the "overfitting" pathology of polynomial regression. A cloud of
noisy samples drawn from a quadratic relationship is fit with an absurdly
high-degree (degree=100) polynomial via a
``PolynomialFeatures + StandardScaler + LinearRegression`` pipeline. The
resulting red model curve wildly oscillates through the samples and blows up
near the boundaries (x -> +-3), shooting out of the visible frame. A
hand-placed violet "prediction" point sits in the lower-right, far from the
data trend, illustrating that an overfit model generalizes poorly and makes
unreliable predictions.

Style follows the shared _style module (clean modern look, CJK fonts, unified
palette). Data reproduces the source notebook: seed 666, uniform x in
[-3, 3], y = 0.5 x^2 + x + 2 + Gaussian noise. The red curve is drawn by
densely sampling np.linspace(-3, 3, N) and predicting, so the oscillation is
faithful (not a sorted-sample polyline). The y-axis is clipped to [0, 10] so
the divergent tails are cropped, exactly as in the original.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import warnings

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import _style
from _style import plt

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "Polynomial-Regression-and-Model-Generalization_1.png"
)

# --- Colors -----------------------------------------------------------------
POINT_COLOR = _style.PALETTE[0]    # blue  #2563EB  -> sample data
CURVE_COLOR = _style.PALETTE[1]    # red   #EF4444  -> overfit model curve
PRED_COLOR = _style.PALETTE[4]     # violet #8B5CF6 -> off-trend prediction point
ANNOT_COLOR = "#444444"


def polynomial_regression(degree):
    """Pipeline: PolynomialFeatures(degree) -> StandardScaler -> LinearReg."""
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    ])


def main():
    # --- Synthetic data: quadratic trend + Gaussian noise (seed 666) --------
    np.random.seed(666)
    x = np.random.uniform(-3.0, 3.0, size=100)
    X = x.reshape(-1, 1)
    y = 0.5 * x**2 + x + 2.0 + np.random.normal(0, 1, size=100)

    # --- Fit a wildly over-parameterized degree-100 polynomial --------------
    # degree=100 on standardized features is numerically explosive; silence
    # the expected ill-conditioning / overflow warnings from the fit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = polynomial_regression(degree=100)
        model.fit(X, y)

        # Dense sampling so the true oscillating curve is captured (not a
        # sorted-sample polyline that would miss the wiggles between points).
        x_plot = np.linspace(-3.0, 3.0, 500).reshape(-1, 1)
        y_plot = model.predict(x_plot)

    fig, ax = _style.new_ax(figsize=(7.2, 5.0))

    # --- Sample scatter (blue) ----------------------------------------------
    ax.scatter(
        x, y, s=46, c=POINT_COLOR, marker="o",
        edgecolors="white", linewidths=0.7, alpha=0.92, zorder=3,
        label="样本数据",
    )

    # --- Overfit model curve (red), densely sampled -------------------------
    ax.plot(
        x_plot.ravel(), y_plot, color=CURVE_COLOR, linewidth=1.7,
        zorder=4, label="过拟合模型 (degree=100)",
    )

    # --- Hand-placed off-trend prediction point (violet) --------------------
    # Conceptual annotation: an overfit model's prediction at x~=2.5 lands far
    # below the local quadratic trend (which would sit near y~=8 there).
    px, py = 2.5, 0.4
    ax.scatter(
        [px], [py], s=320, c=PRED_COLOR, marker="o",
        edgecolors="white", linewidths=1.6, zorder=6, label="预测点",
    )

    # --- Axes ranges & ticks (clip divergent tails, match original) ---------
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_xticks(range(-3, 4))
    ax.set_yticks(range(0, 11, 2))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("过拟合的多项式回归模型 (degree=100)",
                 fontsize=15, color=CURVE_COLOR, pad=12)

    # --- Callout 1: prediction point deviates from the trend ----------------
    ax.annotate(
        "预测值偏离样本趋势",
        xy=(px - 0.08, py + 0.18),
        xytext=(0.55, 1.5),
        fontsize=11, color=PRED_COLOR, ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=PRED_COLOR, linewidth=1.6,
                        connectionstyle="arc3,rad=-0.2"),
        zorder=7,
    )

    # --- Callout 2: curve over-bends to fit every point ---------------------
    # Point to a visible wiggle of the red curve in the mid region.
    wiggle_idx = np.argmin(np.abs(x_plot.ravel() - (-1.2)))
    ax.annotate(
        "为拟合所有点曲线过度弯曲",
        xy=(x_plot.ravel()[wiggle_idx], y_plot[wiggle_idx]),
        xytext=(-0.4, 8.7),
        fontsize=11, color=ANNOT_COLOR, ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color=ANNOT_COLOR, linewidth=1.4,
                        connectionstyle="arc3,rad=0.25"),
        zorder=7,
    )

    ax.legend(loc="upper left", framealpha=0.92)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
