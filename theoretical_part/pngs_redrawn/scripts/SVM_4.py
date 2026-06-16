"""重绘: SVM_4

正面示例：「居中」的决策边界，与两侧数据点距离大致相等。
作为 Hard Margin SVM 核心直觉动机。

规格：
- 无标题、无图例、无坐标刻度
- 仅显示带箭头的 x/y 轴
- 蓝色点群（约 7 个）在左下
- 红色点群（约 7 个）在右上
- 一条蓝色对角直线（从左上到右下）穿越两者之间，视觉上居中

几何设计：
  线：y = -x + 10，即 x + y - 10 = 0，法向量模 sqrt(2)
  点到线距离 = |x + y - 10| / sqrt(2)
  蓝色在线下方（x+y < 10），位于左下角；红色在线上方（x+y > 10），位于右上角。
  要求各点距线至少约 1.8（格子单位），即 |x+y-10| >= 1.8*sqrt(2) ≈ 2.55。
  蓝色：x+y <= 7.4；红色：x+y >= 12.6。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_4.png"

BLUE = "#4472C4"        # 蓝色点（类别 0，左下）
RED = "#C0392B"         # 红色点（类别 1，右上）
LINE_COLOR = "#2563EB"  # 蓝色决策边界线


def main():
    # 蓝色点：x+y <= 7.0（比阈值 7.4 稍保守，保证视觉间距充足）
    # 位于左下区域，分布自然不规则
    blue_pts = np.array([
        [1.2, 1.5],   # x+y=2.7
        [2.5, 2.0],   # x+y=4.5
        [3.5, 2.8],   # x+y=6.3
        [1.8, 4.0],   # x+y=5.8
        [4.5, 2.2],   # x+y=6.7
        [0.8, 3.5],   # x+y=4.3
        [3.0, 4.0],   # x+y=7.0  (最近点, dist=(10-7)/sqrt(2)=2.12 ✓)
    ])

    # 红色点：x+y >= 13.0（比阈值 12.6 稍保守，保证视觉间距充足）
    # 位于右上区域，分布自然不规则
    red_pts = np.array([
        [6.5, 8.0],   # x+y=14.5
        [7.5, 7.5],   # x+y=15.0
        [8.5, 7.0],   # x+y=15.5
        [7.0, 6.5],   # x+y=13.5  (注意：要确保 x+y>=13)
        [9.0, 7.8],   # x+y=16.8
        [6.8, 7.2],   # x+y=14.0
        [8.0, 8.5],   # x+y=16.5
    ])
    # 验证最近红色点 (7.0, 6.5): x+y=13.5，dist=(13.5-10)/sqrt(2)=2.47 ✓

    fig, ax = _style.new_ax(figsize=(6.0, 5.0))

    # --- 散点 ----------------------------------------------------------------
    scatter_kw = dict(s=145, edgecolors="white", linewidths=1.4, zorder=3)
    ax.scatter(blue_pts[:, 0], blue_pts[:, 1], color=BLUE, **scatter_kw)
    ax.scatter(red_pts[:, 0], red_pts[:, 1], color=RED, **scatter_kw)

    # --- 轴范围 ---------------------------------------------------------------
    x_lo, x_hi = 0.0, 10.5
    y_lo, y_hi = 0.0, 10.5
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # --- 居中决策边界 ---------------------------------------------------------
    # 线：y = -x + 10，从左上 (0,10) 到右下 (10,0)，对角穿越图中央
    slope = -1.0
    intercept = 10.0
    xs = np.linspace(0, 10, 300)
    ys = slope * xs + intercept
    mask = (ys >= y_lo) & (ys <= y_hi) & (xs >= x_lo) & (xs <= x_hi)
    ax.plot(
        xs[mask], ys[mask],
        color=LINE_COLOR, linewidth=2.5,
        solid_capstyle="round", zorder=2,
    )

    # --- 手绘风坐标轴：无刻度，仅带箭头的轴线 --------------------------------
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    arrow_kw = dict(
        arrowstyle="-|>",
        color="#333333",
        linewidth=1.6,
        mutation_scale=20,
    )
    ax.annotate("", xy=(x_hi, y_lo), xytext=(x_lo, y_lo), arrowprops=arrow_kw, zorder=1)
    ax.annotate("", xy=(x_lo, y_hi), xytext=(x_lo, y_lo), arrowprops=arrow_kw, zorder=1)

    # --- 输出 -----------------------------------------------------------------
    fig.tight_layout()
    fig.savefig(OUT_PATH, format="png", dpi=200, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
