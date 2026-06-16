"""重绘: Classification-Performance-Measures_3

一维 score 示意图: 横轴是分类器决策函数值 score (θ^T·x_b), 所有样本铺在同一高度
(y=0.02) 的一条水平线上。真实类别 0 用星号、类别 1 用加号区分; 三条竖直线表示
三个不同的决策阈值 (Decision Boundary-1/2/3)。

本帧 (_3) 突出最左侧的 Decision Boundary-3 (score≈-0.019):
  - 此阈值把全部 6 个真实正例都判为 1 -> 召回率 = 6/6 = 1.0
  - 判为 1 的样本共 8 个, 其中 6 个真为 1, 2 个为假阳性 -> 精准率 = 6/8 = 0.75
用来直观说明: 把阈值左移会同时拉高召回率、压低精准率 (precision-recall 权衡)。

点坐标手工合成 (语义沿用 _pic_gen.ipynb), 校准为: 类别 0 整体偏左、类别 1 整体
偏右且有少量重叠交叉, 并满足上面 6/6、6/8 的计数。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Classification-Performance-Measures_3.png"

# --- 手工合成的一维 score 点 (非真实数据集) -------------------------------
# 真实类别 0 (负例) 共 6 个, 整体偏左; 其中 -0.018、-0.009 落在 Boundary-3 右侧,
# 形成 2 个假阳性 (FP=2)。
points_0 = [-0.04, -0.035, -0.03, -0.02, -0.018, -0.009]
# 真实类别 1 (正例) 共 6 个, 整体偏右; 全部落在 Boundary-3 右侧 -> TP=6 (召回率 1.0)。
points_1 = [0.04, 0.03, 0.02, 0.01, 0.008, -0.015]

Y_LEVEL = 0.02       # 所有样本共用的高度 (无实际含义, 仅占位)。
Y_TOP = 0.5          # 竖线高度上限。

# 三条阈值的位置: 编号与位置一致 (1 在 0、2 偏右、3 最左)。
BOUNDARY_1 = 0.0      # Decision Boundary-1
BOUNDARY_2 = 0.025    # Decision Boundary-2
BOUNDARY_3 = -0.019   # Decision Boundary-3 (本帧强调, 最靠左)


def main():
    fig, ax = _style.new_ax()

    # 未强调的两条阈值: 弱化 (细、半透明、虚线)。
    ax.axvline(
        BOUNDARY_1,
        color=_style.PALETTE[0],
        linewidth=1.4,
        linestyle="--",
        alpha=0.45,
        label="决策边界-1 (Decision Boundary-1)",
        zorder=1,
    )
    ax.axvline(
        BOUNDARY_2,
        color=_style.PALETTE[1],
        linewidth=1.4,
        linestyle="--",
        alpha=0.45,
        label="决策边界-2 (Decision Boundary-2)",
        zorder=1,
    )
    # 本帧强调的阈值 Boundary-3: 加粗、高亮 (绿色实线)。
    ax.axvline(
        BOUNDARY_3,
        color=_style.PALETTE[2],
        linewidth=3.0,
        label="决策边界-3 (Decision Boundary-3, 本帧)",
        zorder=2,
    )

    # 类别 0: 蓝色星号; 类别 1: 橙色加号 (沿用原图配色语义)。
    ax.scatter(
        points_0,
        np.full(len(points_0), Y_LEVEL),
        marker="*",
        s=170,
        color=_style.PALETTE[0],
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
        label="类别 0 (真实负例)",
    )
    ax.scatter(
        points_1,
        np.full(len(points_1), Y_LEVEL),
        marker="+",
        s=170,
        linewidths=2.4,
        color=_style.PALETTE[3],
        zorder=4,
        label="类别 1 (真实正例)",
    )

    # 以 Boundary-3 为界的判定方向文字 + 箭头标注 (原创讲解增强)。
    ax.text(
        -0.034,
        0.40,
        "← 预测为 0",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#475569",
    )
    ax.annotate(
        "",
        xy=(-0.046, 0.355),
        xytext=(-0.021, 0.355),
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4),
    )
    ax.text(
        0.008,
        0.40,
        "预测为 1 →",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#475569",
    )
    ax.annotate(
        "",
        xy=(0.034, 0.355),
        xytext=(-0.017, 0.355),
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4),
    )

    # 关键数值说明 (与正文一致: Boundary-3 -> recall=1.0, precision=0.75)。
    ax.text(
        -0.047,
        0.20,
        "决策边界-3 (阈值 score≈-0.019):\n召回率 = 6/6 = 1.0\n精准率 = 6/8 = 0.75",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ECFDF5", ec="#10B981", lw=1.0),
    )

    ax.set_xlabel("分数 score")
    ax.set_title("左移阈值 (决策边界-3): 召回率升、精准率降")

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
