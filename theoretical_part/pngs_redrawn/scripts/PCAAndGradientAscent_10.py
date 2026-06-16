"""Redraw of PCAAndGradientAscent_10: PCA dimensionality reduction before/after.

Scatter comparison of synthetic 2D data before and after a 1-component PCA
round-trip:

  * blue, semi-transparent points  -> original 2D data X (noisy band, slope ~0.75)
  * red, semi-transparent points   -> X_m after transform + inverse_transform;
                                      every red point lands on the first PC axis
  * grey projection lines          -> connect each original point to its
                                      projection, making the information-loss
                                      concept concrete
  * dashed red line                -> first principal component axis guide

Style inherited from _style.py (CJK fonts, white bg, unified palette).
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
from sklearn.decomposition import PCA

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "PCAAndGradientAscent_10.png"
)

BLUE = _style.PALETTE[0]   # #2563EB
RED  = _style.PALETTE[1]   # #EF4444
GREY = "#94A3B8"


def make_data():
    np.random.seed(42)
    X = np.empty((100, 2))
    X[:, 0] = np.random.uniform(0.0, 100.0, size=100)
    X[:, 1] = 0.75 * X[:, 0] + 3.0 + np.random.normal(0.0, 10.0, size=100)
    return X


def pca_roundtrip(X):
    pca = PCA(n_components=1)
    X_m = pca.inverse_transform(pca.fit_transform(X))
    return X_m


def main():
    X   = make_data()
    X_m = pca_roundtrip(X)

    fig, ax = _style.new_ax(figsize=(6.2, 5.0))

    # Grey projection lines (original -> projection)
    for i in range(len(X)):
        ax.plot(
            [X[i, 0], X_m[i, 0]],
            [X[i, 1], X_m[i, 1]],
            color=GREY, linewidth=0.55, alpha=0.45, zorder=1,
        )

    # Blue: original data
    ax.scatter(
        X[:, 0], X[:, 1],
        color=BLUE, alpha=0.5, s=38,
        edgecolors="none", label="$X$（原始数据）",
        zorder=3,
    )

    # Red: PCA-restored data (collinear on PC1 axis)
    ax.scatter(
        X_m[:, 0], X_m[:, 1],
        color=RED, alpha=0.5, s=38,
        edgecolors="none", label="$X_m$（PCA 恢复数据）",
        zorder=4,
    )

    # Dashed PC1 axis guide
    order = np.argsort(X_m[:, 0])
    p0, p1 = X_m[order][0], X_m[order][-1]
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    ext = 18.0
    ax.plot(
        [p0[0] - d[0] * ext, p1[0] + d[0] * ext],
        [p0[1] - d[1] * ext, p1[1] + d[1] * ext],
        color=RED, linestyle="--", linewidth=1.2, alpha=0.55,
        zorder=2, label="第一主成分轴",
    )

    # Annotation box
    ax.text(
        0.03, 0.97,
        "降维后恢复的点仅保留\n第一主成分方向的信息",
        transform=ax.transAxes,
        fontsize=9, va="top", color="#444444",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.88),
    )

    ax.set_xlim(-5, 105)
    ax.set_ylim(-15, 105)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc="lower right", fontsize=9)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
