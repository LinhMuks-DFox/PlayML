"""重绘: SVM_3

二维平面概念示意图, 说明 "决策边界离某一类别过近导致泛化能力差" 的问题:

  - 左下方一簇蓝色训练样本 (蓝色类别)
  - 右上方一簇红色训练样本 (红色类别)
  - 中间偏上、靠近红色簇一侧的一小组绿色测试点
  - 一条从左上走向右下的负斜率蓝色直线作为决策边界, 该边界明显偏向
    红色簇一侧, 使本应更接近红色的绿色测试点落到了直线的蓝色一侧 ->
    发生误分类, 体现泛化能力差。

风格延续系列 (kNN_1 等概念图): 大号实心圆点、无刻度数值、左/底坐标轴
带箭头的极简手绘风。决策边界为手工设定的直线 y = k*x + b, 不做任何拟合,
以保证能画出原图想表达的 "坏边界"。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_3.png"

# 三色语义: 蓝=蓝色类别, 红=红色类别, 绿=待分类/测试样本。
BLUE = _style.PALETTE[0]   # #2563EB
RED = _style.PALETTE[1]    # #EF4444
GREEN = _style.PALETTE[2]  # #10B981

# 决策边界 (手工设定的负斜率直线): y = K * x + B
K = -0.95
B = 9.3


def line_y(x):
    """决策边界直线方程。"""
    return K * x + B


def build_points():
    """手工指定三簇点坐标 (不来自任何数据集), 关键在布局而非数据真实性。"""
    # 蓝色训练样本: 左下方一簇 (约 6 个), 全部落在直线下方 (蓝色一侧)。
    blue = np.array(
        [
            [1.2, 2.3],
            [2.0, 1.2],
            [2.6, 3.0],
            [3.3, 1.7],
            [3.9, 2.6],
            [4.6, 1.5],
        ]
    )
    # 红色训练样本: 右上方一簇 (约 6 个), 全部落在直线上方 (红色一侧)。
    red = np.array(
        [
            [5.6, 7.0],
            [6.4, 5.7],
            [7.2, 7.6],
            [8.0, 6.3],
            [8.6, 7.9],
            [7.6, 5.0],
        ]
    )
    # 绿色测试点: 中间偏上、紧贴红色簇下沿的一小簇 (约 4 个)。
    # 视觉上更靠近红色簇, 但由于边界偏红, 它们落到了直线下方 (蓝色一侧) -> 误分类。
    green = np.array(
        [
            [4.7, 5.6],
            [5.3, 6.3],
            [5.0, 4.9],
            [5.7, 5.4],
        ]
    )
    return blue, red, green


def main():
    blue, red, green = build_points()

    fig, ax = _style.new_ax(figsize=(7, 5.0))

    # --- 决策边界 (负斜率蓝色直线, 贯穿全图) -------------------------------
    x_line = np.array([1.6, 9.2])
    ax.plot(
        x_line,
        line_y(x_line),
        color=BLUE,
        linewidth=2.6,
        zorder=2,
        label="决策边界",
    )

    # --- 三簇散点 (大号实心圆, 白边以求 PPT 风的清晰分离) ----------------
    scatter_kw = dict(s=540, marker="o", edgecolors="white", linewidths=1.6, zorder=3)
    ax.scatter(blue[:, 0], blue[:, 1], c=BLUE, label="蓝色类别", **scatter_kw)
    ax.scatter(red[:, 0], red[:, 1], c=RED, label="红色类别", **scatter_kw)
    ax.scatter(green[:, 0], green[:, 1], c=GREEN, label="测试样本", **scatter_kw)

    # 绘图边界, 给箭头与圆点留白。
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 9.4)

    # 手绘风: 无网格、无刻度数值。
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # 左/底坐标轴 (带箭头)。
    arrow_kw = dict(
        arrowstyle="-|>",
        color="#333333",
        linewidth=1.8,
        mutation_scale=22,
    )
    ax.annotate("", xy=(10.0, 0), xytext=(0, 0), arrowprops=arrow_kw)
    ax.annotate("", xy=(0, 9.4), xytext=(0, 0), arrowprops=arrow_kw)

    # --- 误分类标注: 指向绿色簇, 说明 "实际更接近红色, 却被分到蓝色一侧" ---
    green_center = green.mean(axis=0)
    ax.annotate(
        "实际更接近红色,\n却被分到蓝色一侧",
        xy=(green_center[0] + 0.35, green_center[1] - 0.2),
        xytext=(1.3, 7.6),
        fontsize=12,
        color=GREEN,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6),
        zorder=4,
    )

    # 决策边界已由图例标注, 此处不再重复内嵌文字, 避免与绿色标注/图例重叠。

    ax.set_title("决策边界过于偏向红色类别 → 泛化能力差", color=RED, fontsize=16, pad=14)

    ax.legend(loc="lower right", framealpha=0.9)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
