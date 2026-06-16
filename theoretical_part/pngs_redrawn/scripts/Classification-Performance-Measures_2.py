"""重绘: Classification-Performance-Measures_2

一维分数 (score) 轴上的决策阈值示意图。每个样本被映射到一个标量决策分数
theta^T·x 上 (沿同一条水平线散布), 用两种 marker 区分真实类别 (类0=星号、
类1=加号); 两条垂直线代表两个不同位置的决策阈值 (Decision Boundary)。

本图 (_2) 强调把阈值右移 (分数更高才判为 1, 即采用 Decision Boundary-2,
score=0.025 处): 阈值右侧恰好只剩 2 个真实类1样本被判为 1, 全部判对 ->
精准率 Precision = 2/2 = 1.0; 但 6 个类1样本中有 4 个落在阈值左侧被漏判 ->
召回率 Recall = 2/6 ≈ 0.33。直观说明精准率与召回率此消彼长。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Classification-Performance-Measures_2.png"

# 真实类别 0 (星号) 的分数: 全部偏小, 都落在两条阈值左侧。
POINTS_0 = np.array([-0.040, -0.035, -0.030, -0.020, -0.015, 0.010])
# 真实类别 1 (加号) 的分数: 分散在轴上, 只有最右两个 (0.030, 0.040) 超过
# Decision Boundary-2 (0.025), 其余 4 个落在左侧被漏判。
POINTS_1 = np.array([-0.018, -0.009, 0.008, 0.020, 0.030, 0.040])

# 全部样本固定在同一条水平线上 (y 无实际语义)。
Y_LEVEL = 0.02

# 两条决策阈值。
DB1 = 0.000   # Decision Boundary-1 (较宽松, 蓝色)
DB2 = 0.025   # Decision Boundary-2 (本图采用, 阈值右移, 橙色)


def main():
    fig, ax = _style.new_ax()

    blue = _style.PALETTE[0]    # 蓝: 类0 与 Decision Boundary-1
    orange = _style.PALETTE[3]  # 琥珀/橙: 类1 与 Decision Boundary-2

    # 两条竖直决策边界线 (用 axvline 更稳妥)。
    ax.axvline(
        x=DB1, ymin=0.0, ymax=1.0, color=blue, linewidth=2.0,
        label="Decision Boundary-1",
    )
    ax.axvline(
        x=DB2, ymin=0.0, ymax=1.0, color=orange, linewidth=2.0,
        label="Decision Boundary-2",
    )

    # 两类样本点 (沿同一水平线散布)。
    ax.scatter(
        POINTS_0, np.full_like(POINTS_0, Y_LEVEL),
        marker="*", s=130, color=blue, zorder=5, label="0",
    )
    ax.scatter(
        POINTS_1, np.full_like(POINTS_1, Y_LEVEL),
        marker="+", s=130, linewidths=2.2, color=orange, zorder=5, label="1",
    )

    # 增强标注: 呼应正文中 Precision / Recall 的取值。
    # 阈值 (Decision Boundary-2) 右侧 -> 判为 1; 左侧 -> 判为 0。
    ax.annotate(
        "判为 0", xy=(-0.020, 0.038), ha="center", va="center",
        fontsize=10, color="#555555",
    )
    ax.annotate(
        "判为 1", xy=(0.0375, 0.072), ha="center", va="center",
        fontsize=10, color="#555555",
    )
    ax.annotate(
        "Precision = 2/2 = 1.0\nRecall = 2/6 ≈ 0.33",
        xy=(0.0405, 0.052), ha="right", va="center", fontsize=10.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", lw=1.0),
    )

    ax.set_xlabel("score")
    ax.set_title("决策阈值右移: 精准率 ↑ / 召回率 ↓")

    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(0.0, 0.085)

    # y 轴无实际含义, 隐藏其刻度与标签让示意更干净。
    ax.get_yaxis().set_visible(False)
    ax.grid(False)

    ax.legend(loc="upper left", framealpha=0.9)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
