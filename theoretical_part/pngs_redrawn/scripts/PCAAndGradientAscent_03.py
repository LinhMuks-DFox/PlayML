"""Redraw of PCAAndGradientAscent_03: 丢弃 x 特征、只保留 y 特征。

PCA 动机说明图。使用与 _01 / _02 完全相同的小型演示数据集（7 个点，seed=42），
将 x 特征丢弃，仅保留 y 特征，以样本序号作为横坐标（表示 x 维度已无意义）。
两类点按 y 值中位数二分着色（蓝/红），呈现出 y 方向上可分性较低的状态，
与 _02 图（保留 x 特征、方差大、可分性好）形成对比，说明该投影方向信息损失较大。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/PCAAndGradientAscent_03.png"

# --- Same small demo dataset as _01 and _02 (seed=42, 7 points) ------------
np.random.seed(42)
n = 7
x_pts = np.linspace(1.5, 5.8, n) + np.random.randn(n) * 0.15
y_pts = 0.4 * x_pts + 1.0 + np.random.randn(n) * 0.18
x_pts = np.clip(x_pts, 1.0, 6.2)
y_pts = np.clip(y_pts, 1.5, 3.5)

BLUE = _style.PALETTE[0]   # #2563EB
RED  = _style.PALETTE[1]   # #EF4444


def main():
    # 以 y 值中位数做简单二分，给两类着色（仅为视觉示意，非真实标签）
    y_median = np.median(y_pts)
    mask_blue = y_pts >= y_median
    mask_red  = ~mask_blue

    # 样本序号充当 x 轴（x 维度已被丢弃，该轴无语义）
    idx = np.arange(n)

    # --- Plot ----------------------------------------------------------------
    fig, ax = _style.new_ax(figsize=(6.4, 4.4))

    ax.scatter(
        idx[mask_blue], y_pts[mask_blue],
        s=90, color=BLUE, edgecolors="white", linewidths=1.2, zorder=4,
    )
    ax.scatter(
        idx[mask_red], y_pts[mask_red],
        s=90, color=RED, edgecolors="white", linewidths=1.2, zorder=4,
    )

    # 保持与 _02 相同的 y 轴范围，便于读者对比两图
    ax.set_ylim(-0.35, 3.9)
    ax.set_xlim(-0.5, n - 0.5)

    ax.set_xticks(idx)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    ax.set_xlabel("样本序号（x 特征已丢弃）")
    ax.set_ylabel("特征 y")
    ax.set_title("丢弃 x 特征，保留 y 特征")

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
