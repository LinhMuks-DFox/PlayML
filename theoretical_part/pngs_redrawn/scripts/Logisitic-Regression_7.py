"""重绘: Logisitic-Regression_7

二维非线性可分数据的散点图: 一类 (红色) 聚集在原点附近, 另一类 (蓝色) 环绕
在外侧形成环形区域。用于说明直线决策边界无法分开这种数据, 从而引出
多项式特征 / 圆形决策边界的逻辑回归。这是 "之前" 的图, 不画任何决策边界。

修正 QA 问题:
  1. 点数量缩减至约 60 个, 避免红色区域成实心色块。
  2. 坐标轴范围留出足够 padding, 所有点完整显示在画布内。
  3. 红色点可单独辨识。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Logisitic-Regression_7.png"

# Colors matching the spec
COLOR_0 = '#4FC3F7'  # light blue for outer ring (class 0)
COLOR_1 = '#EF5350'  # red for inner circle (class 1)


def main():
    # 生成少量数据点 (~60 个), 保持内外圈结构清晰可辨。
    # 用较小的 normal std (0.9) 使数据整体更紧凑, 减少离群点超出画面。
    np.random.seed(666)
    X = np.random.normal(0, 0.9, size=(60, 2))
    y = np.array(X[:, 0] ** 2 + X[:, 1] ** 2 < 1.5, dtype="int")

    # 计算数据实际范围并加 padding, 确保没有点被截断。
    pad = 0.55
    x_abs = np.abs(X).max() + pad
    lim = max(x_abs, 2.2)  # 至少 ±2.2 保证原点十字轴美观

    fig, ax = _style.new_ax(figsize=(5.5, 5.5))

    # 外圈 (class 0) 用浅蓝色, 内圈 (class 1) 用红色, 与规格配色一致。
    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color=COLOR_0,
        s=130,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color=COLOR_1,
        s=130,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )

    # 对称的坐标范围, 保证原点居中, 所有点完整显示。
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    # 等比例, 让环形结构看起来是圆而不是椭圆。
    ax.set_aspect("equal")

    # 关闭网格与刻度, 模仿原图的概念示意风格。
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # 隐藏所有外框 spines。
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # 用带箭头的注释画出穿过原点的 x 轴与 y 轴 (十字坐标轴)。
    # 箭头终点略短于 lim, 避免与画布边缘碰撞。
    arrow_end = lim * 0.96
    arrow_kw = dict(arrowstyle="-|>", color="#333333", lw=1.4, mutation_scale=16)
    # x 轴: 从左到右。
    ax.annotate("", xy=(arrow_end, 0), xytext=(-arrow_end, 0),
                arrowprops=arrow_kw, zorder=1, annotation_clip=False)
    # y 轴: 从下到上。
    ax.annotate("", xy=(0, arrow_end), xytext=(0, -arrow_end),
                arrowprops=arrow_kw, zorder=1, annotation_clip=False)

    # 无标题、无图例文字 (与规格一致, 颜色即区分)。

    # 直接保存, 不走 finalize (finalize 会重新开启 top/right spines 之外的处理,
    # 这里已自行处理所有 spines)。
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
