"""重绘: Classification-Performance-Measures_1

一维 score 示意图: 横轴是分类器决策函数值 score (θ^T·x_b), 所有样本铺在同一高度
(y=0.02) 的一条水平线上。真实类别 0 用星号、类别 1 用加号区分; 一条竖直线表示
决策边界 (阈值)。

用来说明: 在 score=0 处划界时
  - 判为 1 (score>0) 的样本共 5 个, 其中 4 个真为 1  -> 精准率 = 4/5 = 0.8
  - 真为 1 的样本共 6 个, 其中 4 个落在边界右侧 -> 召回率 = 4/6 ≈ 0.67
平移决策边界会改变 TP/FP 的划分, 从而让精准率与召回率此消彼长。

点坐标完全手工合成 (沿用 _pic_gen.ipynb cell 1 的 points_0/points_1),
并以正文混淆矩阵数值 (4/5、4/6) 校准。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Classification-Performance-Measures_1.png"

# 手工合成的一维 score 点 (非真实数据集)。
# 真实类别 0: 只有 0.01 落在边界右侧 -> 1 个假阳性 (FP=1)。
points_0 = [-0.04, -0.035, -0.03, -0.02, -0.015, 0.01]
# 真实类别 1: 0.008/0.02/0.03/0.04 在右侧 (TP=4), -0.009/-0.018 在左侧 (FN=2)。
points_1 = [0.04, 0.03, 0.02, 0.008, -0.009, -0.018]

Y_LEVEL = 0.02       # 所有样本共用的高度 (无实际含义, 仅占位)。
BOUNDARY_X = 0.0     # 决策边界 / 阈值。
Y_TOP = 0.5          # 竖线高度上限。


def main():
    fig, ax = _style.new_ax()

    # 决策边界竖线 (放在散点之下, 避免压住样本)。
    ax.axvline(
        BOUNDARY_X,
        color=_style.PALETTE[0],
        linewidth=2.0,
        label="决策边界 (Decision Boundary-1)",
        zorder=1,
    )

    # 类别 0: 蓝色星号; 类别 1: 橙色加号 (沿用原图配色语义)。
    ax.scatter(
        points_0,
        np.full(len(points_0), Y_LEVEL),
        marker="*",
        s=160,
        color=_style.PALETTE[0],
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
        label="类别 0 (真实)",
    )
    ax.scatter(
        points_1,
        np.full(len(points_1), Y_LEVEL),
        marker="+",
        s=160,
        linewidths=2.2,
        color=_style.PALETTE[3],
        zorder=3,
        label="类别 1 (真实)",
    )

    # 左右两侧的判定方向文字标注 (原创讲解增强)。
    ax.text(
        -0.022,
        0.32,
        "判为 0",
        ha="center",
        va="center",
        fontsize=12,
        color="#475569",
    )
    ax.annotate(
        "",
        xy=(-0.038, 0.27),
        xytext=(-0.006, 0.27),
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4),
    )
    ax.text(
        0.022,
        0.32,
        "判为 1",
        ha="center",
        va="center",
        fontsize=12,
        color="#475569",
    )
    ax.annotate(
        "",
        xy=(0.038, 0.27),
        xytext=(0.006, 0.27),
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4),
    )

    # 关键数值说明 (与正文混淆矩阵一致)。放在左下方空白区, 避免压住图例与散点。
    ax.text(
        -0.047,
        0.12,
        "精准率 = 4/5 = 0.80\n召回率 = 4/6 ≈ 0.67",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.4", fc="#F1F5F9", ec="#CBD5E1", lw=0.8),
    )

    ax.set_xlabel("分数 score")
    ax.set_title("决策边界平移如何改变精准率与召回率")

    # y 轴无实际含义: 隐藏刻度, 仅给竖线留高度。
    ax.set_ylim(-0.02, Y_TOP + 0.03)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.spines["left"].set_visible(False)
    ax.grid(False)

    ax.set_xlim(-0.05, 0.05)

    ax.legend(loc="upper right", framealpha=0.95)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
