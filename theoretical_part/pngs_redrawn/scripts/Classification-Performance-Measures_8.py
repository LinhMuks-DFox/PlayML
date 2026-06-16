"""重绘: Classification-Performance-Measures_8

一维 "score 轴" 分类示意图(图 7-10 系列的第 2 张)。12 个一维样本点沿 score
横轴排列(纵坐标固定在 y=0.02 处只为可见), 一条竖直的 "Decision Boundary-2"
落在 score=0.0 处。用于讲解 ROC: 在该阈值下, 边界右侧(score>0)判为 1 的点中,
类 0 仅 0.01 一点 → FP=1 → FPR = 1/6 ≈ 0.16; 类 1 有 {0.008, 0.02, 0.03, 0.04}
四点 → TP=4 → TPR = 4/6 ≈ 0.67。

坐标与 _7 共用同一组合成点, 仅移动决策边界竖线的 x 位置 (0.025 → 0.0)。
坐标为作者硬编码的合成值, 务必沿用以保证 FPR/TPR 与正文一致。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Classification-Performance-Measures_8.png"

# 作者固定坐标(合成手工值): 与 _7 完全一致, 复现时不可改动。
POINTS_0 = [-0.04, -0.03, -0.02, -0.015, 0.01, -0.035]  # 类别 0
POINTS_1 = [0.04, 0.03, 0.02, 0.008, -0.009, -0.018]    # 类别 1
Y_LEVEL = 0.02                                          # 所有点抬到的可见高度
BOUNDARY_X = 0.0                                        # Decision Boundary-2


def main():
    blue = _style.PALETTE[0]    # 类别 0 + 决策边界
    orange = _style.PALETTE[3]  # 类别 1 (amber/orange)

    fig, ax = _style.new_ax()

    # 竖直决策边界线: 从 y=0 画到 y=0.5。
    ax.plot(
        [BOUNDARY_X, BOUNDARY_X],
        [0.0, 0.5],
        color=blue,
        linewidth=2.0,
        label="Decision Boundary-2",
    )

    # 类别 0 样本: 蓝色星号。
    ax.scatter(
        POINTS_0,
        np.full(len(POINTS_0), Y_LEVEL),
        marker="*",
        s=130,
        color=blue,
        zorder=5,
        label="0",
    )
    # 类别 1 样本: 橙色加号。
    ax.scatter(
        POINTS_1,
        np.full(len(POINTS_1), Y_LEVEL),
        marker="+",
        s=130,
        linewidths=2.2,
        color=orange,
        zorder=5,
        label="1",
    )

    # 边界两侧 "判为 1 / 判为 0" 区域提示。
    ax.annotate(
        "判为 1 (score>0)",
        xy=(BOUNDARY_X + 0.004, 0.34),
        ha="left",
        va="bottom",
        fontsize=11,
        color=blue,
    )
    ax.annotate(
        "判为 0",
        xy=(BOUNDARY_X - 0.004, 0.46),
        ha="right",
        va="bottom",
        fontsize=11,
        color="#64748B",
    )

    # ROC 数值标注, 呼应正文。
    ax.text(
        0.27,
        0.7,
        "FPR = 1/6 ≈ 0.16\nTPR = 4/6 ≈ 0.67",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#CCCCCC",
        ),
    )

    ax.set_xlabel("score")
    # y 轴无实际语义, 仅用于把点抬到可见高度 → 隐藏刻度避免误解。
    ax.set_yticks([])
    ax.set_ylabel("")

    ax.set_ylim(-0.02, 0.55)
    ax.set_xlim(-0.052, 0.052)

    ax.legend(loc="upper right")

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
