"""重绘: SVM_8

SVM margin 概念示意图 — 展示两类线性可分散点、三条平行斜线 (A/B/C)
以及双头箭头标注的间距 d，说明 margin = 2d 的核心思想。

  - 约 10 个红色实心圆点 (正类，右上方) 全部落在上 margin 线 A 之上。
  - 约 10 个蓝色实心圆点 (负类，左下方) 全部落在下 margin 线 B 之下。
  - 三条斜率相同的平行直线（手工设定，不拟合真实模型）:
      A — 上边界 (上 margin)
      C — 中间决策边界
      B — 下边界 (下 margin)
  - 3 个支撑向量 (落在 A/B 线上) 用黑色描边高亮。
  - 两段双头箭头标注 C 到 A / C 到 B 的正交距离，各旁边文字 'd'。
  - 右下方文字 'margin = 2d'。
  - 坐标轴只显示正半轴 (带箭头)，无刻度，无网格。

风格与 SVM_7 系列完全一致（斜率 K = -1，截距 B0 = 9，半间距 GAP = 1.9）。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_8.png"

BLUE = _style.PALETTE[0]    # #2563EB  -> 负类，左下
RED  = _style.PALETTE[1]    # #EF4444  -> 正类，右上
LINE_COLOR = "#3730A3"      # 深靛蓝，与原图蓝紫斜线呼应

# --- 三条平行直线参数（与 SVM_7 完全一致，保证系列统一）-------------------
K   = -1.0    # 公共斜率
B0  = 9.0     # 决策边界 C 的截距
GAP = 1.9     # 上/下 margin 线与决策边界的截距差


def line_y(x, b):
    """y = K*x + b"""
    return K * x + b


def main():
    # -----------------------------------------------------------------------
    # 手工指定散点坐标，确保线性可分：
    #   正类 (红): y > K*x + (B0 + GAP)，即在 A 线以上（上方区域）
    #   负类 (蓝): y < K*x + (B0 - GAP)，即在 B 线以下（下方区域）
    # -----------------------------------------------------------------------
    red = np.array([
        [4.8, 7.8],
        [5.8, 8.5],
        [6.5, 7.2],
        [7.2, 8.0],
        [8.0, 6.8],
        [5.5, 5.8],
        [6.8, 6.0],
        [7.8, 5.5],
        [8.6, 7.5],
        [9.0, 6.2],
    ])

    blue = np.array([
        [1.2, 3.0],
        [2.0, 1.8],
        [2.8, 3.6],
        [3.5, 1.5],
        [2.5, 4.5],
        [4.2, 2.2],
        [1.5, 5.0],
        [3.8, 3.2],
        [4.5, 1.0],
        [1.0, 1.5],
    ])

    # --- 支撑向量: 精确落在 margin 线上 ------------------------------------
    # 2 个红色支撑向量落在上 margin 线 A (截距 = B0 + GAP)
    sv_red_x  = np.array([4.0, 6.0])
    sv_red    = np.column_stack([sv_red_x, line_y(sv_red_x, B0 + GAP)])
    # 1 个蓝色支撑向量落在下 margin 线 B (截距 = B0 - GAP)
    sv_blue_x = np.array([5.2])
    sv_blue   = np.column_stack([sv_blue_x, line_y(sv_blue_x, B0 - GAP)])

    # -----------------------------------------------------------------------
    fig, ax = _style.new_ax(figsize=(7.0, 5.8))

    x_lo, x_hi = 0.0, 10.8
    y_lo, y_hi = 0.0, 11.0
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # --- 三条平行直线 -------------------------------------------------------
    xs = np.linspace(0.2, 10.4, 400)

    def plot_line(b, lw=2.0):
        ys = line_y(xs, b)
        m  = (ys >= y_lo + 0.1) & (ys <= y_hi - 0.1)
        ax.plot(xs[m], ys[m], color=LINE_COLOR, linewidth=lw,
                solid_capstyle="round", zorder=2)

    plot_line(B0 + GAP, lw=2.0)    # 上边界 A
    plot_line(B0 - GAP, lw=2.0)    # 下边界 B
    plot_line(B0,       lw=3.0)    # 决策边界 C（稍粗）

    # --- 字母标注 A / B / C -----------------------------------------------
    # 在直线左端附近（x ≈ 0.8）偏左标注，避免与点重叠
    label_font = dict(fontsize=20, fontweight="bold", color="#1e1b4b",
                      ha="right", va="center", zorder=6)
    x_lbl = 0.8
    ax.text(x_lbl - 0.25, line_y(x_lbl, B0 + GAP), "A", **label_font)
    ax.text(x_lbl - 0.25, line_y(x_lbl, B0),       "C", **label_font)
    ax.text(x_lbl - 0.25, line_y(x_lbl, B0 - GAP), "B", **label_font)

    # --- 散点（大号实心圆，白边）------------------------------------------
    scatter_kw = dict(s=220, marker="o", edgecolors="white",
                      linewidths=1.2, zorder=3)
    ax.scatter(red[:, 0],  red[:, 1],  c=RED,  **scatter_kw)
    ax.scatter(blue[:, 0], blue[:, 1], c=BLUE, **scatter_kw)

    # --- 支撑向量高亮（黑色描边）------------------------------------------
    sv_kw = dict(s=280, marker="o", edgecolors="black",
                 linewidths=2.4, zorder=5)
    ax.scatter(sv_red[:, 0],  sv_red[:, 1],  c=RED,  **sv_kw)
    ax.scatter(sv_blue[:, 0], sv_blue[:, 1], c=BLUE, **sv_kw)

    # -----------------------------------------------------------------------
    # 距离标注: 正交（垂直于斜线）的双头箭头 + 'd'
    # 斜线方向向量: (1, K) = (1, -1)，单位法向量: (K, -1) 归一化
    # 法向量（垂直于线，指向上方）: n̂ = (−K, 1) / sqrt(1 + K²)
    # -----------------------------------------------------------------------
    norm_factor = np.sqrt(1 + K**2)
    nx, ny = -K / norm_factor, 1.0 / norm_factor   # 单位法向量（指向截距增大方向）
    # GAP 为截距差，正交距离 = GAP / sqrt(1 + K²) = GAP / norm_factor
    perp_dist = GAP / norm_factor

    # 箭头锚点：选取 x=7.8 处的 C 线上一点，再沿法向量延伸到 A/B
    x_anchor = 7.8
    cx = x_anchor
    cy = line_y(cx, B0)
    ax_pt  = np.array([cx + nx * perp_dist, cy + ny * perp_dist])   # A 线上的点
    bx_pt  = np.array([cx - nx * perp_dist, cy - ny * perp_dist])   # B 线上的点
    c_pt   = np.array([cx, cy])

    arrow_props = dict(
        arrowstyle="<->",
        color="black",
        linewidth=1.8,
        mutation_scale=16,
        shrinkA=0,
        shrinkB=0,
    )

    # C → A 段（双头箭头）
    ax.annotate(
        "",
        xy=(ax_pt[0], ax_pt[1]),
        xytext=(c_pt[0], c_pt[1]),
        arrowprops=arrow_props,
        zorder=4,
    )
    # 标注 'd'（偏右上，沿斜线方向偏移）
    mid_ca = (c_pt + ax_pt) / 2
    ax.text(mid_ca[0] + 0.35, mid_ca[1] + 0.0,
            "d", fontsize=17, fontweight="bold", color="black",
            ha="left", va="center", zorder=6)

    # C → B 段（双头箭头）
    ax.annotate(
        "",
        xy=(bx_pt[0], bx_pt[1]),
        xytext=(c_pt[0], c_pt[1]),
        arrowprops=arrow_props,
        zorder=4,
    )
    mid_cb = (c_pt + bx_pt) / 2
    ax.text(mid_cb[0] + 0.35, mid_cb[1] + 0.0,
            "d", fontsize=17, fontweight="bold", color="black",
            ha="left", va="center", zorder=6)

    # 'margin = 2d' 文字（B 线右下方）
    ax.text(
        bx_pt[0] + 0.3, bx_pt[1] - 0.45,
        "margin = 2d",
        fontsize=13, color="black", style="italic",
        ha="left", va="top", zorder=6,
    )

    # -----------------------------------------------------------------------
    # 坐标轴: 仅显示正半轴，带箭头，无刻度，无网格
    # -----------------------------------------------------------------------
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    arrow_axis_kw = dict(
        arrowstyle="-|>", color="#333333",
        linewidth=1.8, mutation_scale=20,
    )
    ax.annotate("", xy=(x_hi, 0), xytext=(0, 0),
                arrowprops=arrow_axis_kw, zorder=1)
    ax.annotate("", xy=(0, y_hi), xytext=(0, 0),
                arrowprops=arrow_axis_kw, zorder=1)

    ax.set_title("SVM 的间隔: margin = 2d", fontsize=16, pad=12)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="png", dpi=200, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
