"""重绘: SVM_7

二维平面 SVM 支撑向量 (Support Vector) 概念示意图:

  - 两类线性可分的散点: 红色一类在右上方区域, 蓝色一类在左下方区域。
  - 三根互相平行的负斜率斜线 (从左上到右下): 中间一根为决策边界,
    上下两根为 margin 边界。三线严格平行 (同一斜率 K)。
  - 高亮 3 个支撑向量: 2 个红点 + 1 个蓝点, 恰好落在上下两条 margin
    边界上 (红点在上 margin 线, 蓝点在下 margin 线), 用更大的点 + 黑色
    描边突出。
  - 从这 3 个支撑向量出发的粗黑连接线/箭头, 汇聚指向右侧的 "SV" 文字标注,
    说明正是这三个点定义了整个 margin 区域与决策边界。

风格延续系列 (SVM_3/SVM_4 等概念图): 大号实心圆点、无刻度数值、左/底坐标轴
带箭头的极简手绘风, 红=正类/蓝=负类 配色与同组一致。三条线为手工设定的
平行直线 (不做拟合), 支撑向量点坐标精确落在 margin 线上以保证几何关系正确。
与 SVM_6 (两条 margin 平行线)、SVM_8 (margin 距离 d 标注) 同属一组, 坐标系、
配色、斜率统一。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_7.png"

BLUE = _style.PALETTE[0]   # #2563EB  -> 负类, 左下
RED = _style.PALETTE[1]    # #EF4444  -> 正类, 右上
LINE_COLOR = "#6D28D9"     # 三条平行线的紫蓝色 (与原图蓝紫斜线呼应)

# --- 三条平行直线 (统一斜率 K, 决策边界截距 B0, margin 半间距 GAP) -------
# 直线方程: y = K * x + B
#   决策边界:   y = K*x + B0
#   上 margin:  y = K*x + B0 + GAP   (红色支撑向量落在此线)
#   下 margin:  y = K*x + B0 - GAP   (蓝色支撑向量落在此线)
K = -1.0
B0 = 9.0
GAP = 1.9


def line_y(x, b):
    """平行直线方程 y = K*x + b。"""
    return K * x + b


def main():
    # --- 两类散点 (手工指定坐标, 关键在布局而非数据真实性) ----------------
    # 红色 (正类): 右上方区域, 全部落在上 margin 线之上 (margin 区域外)。
    red = np.array(
        [
            [4.2, 7.6],
            [5.6, 8.3],
            [6.6, 7.0],
            [7.4, 8.1],
            [8.0, 6.4],
            [6.0, 5.6],
        ]
    )
    # 蓝色 (负类): 左下方区域, 全部落在下 margin 线之下 (margin 区域外)。
    blue = np.array(
        [
            [1.3, 3.2],
            [2.2, 1.9],
            [3.0, 3.4],
            [3.6, 1.6],
            [2.6, 4.4],
            [4.3, 2.3],
        ]
    )

    # --- 3 个支撑向量: 坐标精确落在 margin 线上 ----------------------------
    # 2 个红色支撑向量落在上 margin 线 y = K*x + (B0 + GAP)。
    sv_red_x = np.array([4.0, 5.6])
    sv_red = np.column_stack([sv_red_x, line_y(sv_red_x, B0 + GAP)])
    # 1 个蓝色支撑向量落在下 margin 线 y = K*x + (B0 - GAP)。
    sv_blue_x = np.array([5.2])
    sv_blue = np.column_stack([sv_blue_x, line_y(sv_blue_x, B0 - GAP)])
    support_vectors = np.vstack([sv_red, sv_blue])

    fig, ax = _style.new_ax(figsize=(7.0, 5.2))

    # --- 视图范围 ----------------------------------------------------------
    x_lo, x_hi = 0.0, 10.4
    y_lo, y_hi = 0.0, 10.0
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # --- 三条平行直线 ------------------------------------------------------
    xs = np.linspace(0.4, 9.6, 200)
    # 上下 margin 边界 (较细, 与决策边界区分)。
    for b in (B0 + GAP, B0 - GAP):
        ys = line_y(xs, b)
        m = (ys >= y_lo + 0.2) & (ys <= y_hi - 0.2)
        ax.plot(
            xs[m], ys[m], color=LINE_COLOR, linewidth=1.8,
            linestyle="--", solid_capstyle="round", zorder=2,
        )
    # 中间决策边界 (较粗实线)。
    ys0 = line_y(xs, B0)
    m0 = (ys0 >= y_lo + 0.2) & (ys0 <= y_hi - 0.2)
    ax.plot(
        xs[m0], ys0[m0], color=LINE_COLOR, linewidth=3.0,
        solid_capstyle="round", zorder=2,
    )

    # --- 两类散点 (大号实心圆, 白边) ---------------------------------------
    scatter_kw = dict(s=300, marker="o", edgecolors="white",
                      linewidths=1.4, zorder=3)
    ax.scatter(red[:, 0], red[:, 1], c=RED, label="正类 (红)", **scatter_kw)
    ax.scatter(blue[:, 0], blue[:, 1], c=BLUE, label="负类 (蓝)", **scatter_kw)

    # --- 高亮 3 个支撑向量 (更大的点 + 黑色粗描边) -------------------------
    sv_kw = dict(s=360, marker="o", edgecolors="black", linewidths=2.6, zorder=5)
    ax.scatter(sv_red[:, 0], sv_red[:, 1], c=RED, **sv_kw)
    ax.scatter(sv_blue[:, 0], sv_blue[:, 1], c=BLUE, **sv_kw)

    # --- 从 3 个支撑向量汇聚到右侧 "SV" 标注的粗黑箭头 ---------------------
    sv_label_xy = (9.7, 5.2)   # "SV" 文字附近的汇聚点 (右侧)
    for pt in support_vectors:
        ax.annotate(
            "",
            xy=(pt[0], pt[1]),          # 箭头指向支撑向量
            xytext=sv_label_xy,         # 从 SV 标注一侧出发
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                linewidth=3.0,
                mutation_scale=20,
                shrinkA=2,
                shrinkB=10,
            ),
            zorder=4,
        )
    ax.text(
        sv_label_xy[0] + 0.05, sv_label_xy[1],
        "SV",
        fontsize=22, fontweight="bold", color="black",
        ha="left", va="center", zorder=6,
    )

    # --- 手绘风: 无网格、无刻度数值, 仅左/底带箭头坐标轴 -------------------
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    arrow_kw = dict(arrowstyle="-|>", color="#333333", linewidth=1.8,
                    mutation_scale=22)
    ax.annotate("", xy=(x_hi, 0), xytext=(0, 0), arrowprops=arrow_kw, zorder=1)
    ax.annotate("", xy=(0, y_hi), xytext=(0, 0), arrowprops=arrow_kw, zorder=1)

    ax.set_title("支撑向量 (SV) 定义了 margin 区域与决策边界", fontsize=16, pad=12)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="png", dpi=200, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
