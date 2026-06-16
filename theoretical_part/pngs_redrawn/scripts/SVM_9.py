"""Redraw of SVM_9: the SVM hard-margin geometry concept diagram.

三条平行线：
  C = 上方虚线   w_d^T x + b_d = +1  (红色支撑向量在此)
  B = 中间实线   w^T x + b = 0        (决策边界)
  A = 下方虚线   w_d^T x + b_d = -1  (蓝色支撑向量在此)

修复要点（v15）：
  1. 公式标签：三条公式标签分别精确贴靠各自对应线的右端。
  2. 符号修正：上方虚线公式标签写为 w_d^T x + b_d = +1（含正号）。
  3. 2d 双向箭头（核心修复）：
     - 下端：线 A（w_d^T x + b_d = -1），位于 A 字母标签的正右侧，numpy 断言验证。
     - 上端：线 C（w_d^T x + b_d = +1），numpy 断言验证，跨越完整 2d = margin。
     - 为消除视觉歧义，在箭头与线 B 的交叉点处加横向 tick 标记。
     - 标签 "2d (margin)" 放在 B 线与 C 线之间（箭头左侧）。
  4. d  双向箭头：严格从线 A 到线 B（半 margin = d）。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_9.png"

RED = _style.PALETTE[1]    # #EF4444  -> class +1
BLUE = _style.PALETTE[0]   # #2563EB  -> class -1
LINE_COLOR = "#2B3EBF"
TEXT_COLOR = "#1F2937"

SLOPE = -0.8   # 斜率（线向右下倾斜）
MARGIN = 2.0   # 相邻两线 y 截距差


def line_y(x, c=0.0):
    """y = SLOPE*x + c"""
    return SLOPE * x + c


def main():
    x_lo, x_hi = -1.5, 9.0
    y_lo, y_hi = -9.5, 6.0

    fig, ax = _style.new_ax(figsize=(10.0, 9.0))
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")

    xs = np.linspace(x_lo, x_hi, 500)

    def clip(x_arr, y_arr):
        mask = (y_arr >= y_lo) & (y_arr <= y_hi)
        return x_arr[mask], y_arr[mask]

    # ------------------------------------------------------------------
    # 基础几何量（numpy 精确计算）
    # 法向量 n_hat：垂直于线，朝"从 A 走向 C"方向
    # 切向量 tang： 沿线方向，朝 x 增大方向
    # perp_dist：   相邻两线真实垂直距离 = d
    # ------------------------------------------------------------------
    m = SLOPE
    n_hat = np.array([-m, 1.0]) / np.sqrt(1.0 + m ** 2)
    tang  = np.array([1.0,   m]) / np.sqrt(1.0 + m ** 2)
    perp_dist = MARGIN / np.sqrt(1.0 + m ** 2)

    # ------------------------------------------------------------------
    # 三条线
    # ------------------------------------------------------------------
    xC, yC = clip(xs, line_y(xs, +MARGIN))
    ax.plot(xC, yC, color=LINE_COLOR, lw=2.2, linestyle="--",
            solid_capstyle="round", zorder=2)

    xB, yB = clip(xs, line_y(xs, 0))
    ax.plot(xB, yB, color=LINE_COLOR, lw=3.2, linestyle="-",
            solid_capstyle="round", zorder=2)

    xA, yA = clip(xs, line_y(xs, -MARGIN))
    ax.plot(xA, yA, color=LINE_COLOR, lw=2.2, linestyle="--",
            solid_capstyle="round", zorder=2)

    # ------------------------------------------------------------------
    # A/B/C 字母标签（对齐各自对应线）
    # ------------------------------------------------------------------
    lx = -0.8   # 字母标签 x 锚点（各线在此 x 处的 y 值精确对应）

    ax.text(lx - 0.30, line_y(lx, +MARGIN), "C",
            color=TEXT_COLOR, fontsize=18, fontstyle="italic",
            fontweight="bold", ha="right", va="center", zorder=7)
    ax.text(lx - 0.30, line_y(lx, 0),       "B",
            color=TEXT_COLOR, fontsize=18, fontstyle="italic",
            fontweight="bold", ha="right", va="center", zorder=7)
    ax.text(lx - 0.30, line_y(lx, -MARGIN), "A",
            color=TEXT_COLOR, fontsize=18, fontstyle="italic",
            fontweight="bold", ha="right", va="center", zorder=7)

    # ------------------------------------------------------------------
    # 支撑向量点
    # ------------------------------------------------------------------
    sv_red_x0  = np.array([0.5, 2.5])
    sv_red     = np.column_stack([sv_red_x0,  line_y(sv_red_x0,  +MARGIN)])
    sv_blue_x0 = np.array([2.0])
    sv_blue    = np.column_stack([sv_blue_x0, line_y(sv_blue_x0, -MARGIN)])

    # ------------------------------------------------------------------
    # 普通样本点（在 margin 带之外）
    # ------------------------------------------------------------------
    red_x  = np.array([-1.0, 0.5, 2.0, 4.0, 1.5])
    red_y  = line_y(red_x,  +MARGIN) + np.array([1.5, 1.3, 2.0, 1.2, 2.5])
    blue_x = np.array([-0.5, 1.0, 3.0, 4.5, 2.5])
    blue_y = line_y(blue_x, -MARGIN) - np.array([1.4, 1.0, 1.6, 1.0, 1.8])

    def in_view(xa, ya):
        mask = (xa >= x_lo) & (xa <= x_hi) & (ya >= y_lo) & (ya <= y_hi)
        return xa[mask], ya[mask]

    rx, ry = in_view(red_x, red_y)
    bx, by = in_view(blue_x, blue_y)
    ax.scatter(rx, ry,  color=RED,  s=150, edgecolors="white", linewidths=1.2, zorder=4)
    ax.scatter(bx, by,  color=BLUE, s=150, edgecolors="white", linewidths=1.2, zorder=4)
    ax.scatter(sv_red[:,0],  sv_red[:,1],  color=RED,  s=300,
               edgecolors=TEXT_COLOR, linewidths=2.5, zorder=5)
    ax.scatter(sv_blue[:,0], sv_blue[:,1], color=BLUE, s=300,
               edgecolors=TEXT_COLOR, linewidths=2.5, zorder=5)

    # ==================================================================
    # 距离标注箭头（沿 n_hat 方向，垂直于决策边界）
    #
    #   从线 A 上的点 P 沿 n_hat 走 1×perp_dist  → 到达线 B  （d）
    #   从线 A 上的点 P 沿 n_hat 走 2×perp_dist  → 到达线 C  （2d）
    # ==================================================================

    # ================================================================
    # [1] 2d 双向箭头：线 A → 线 C（完整 margin = 2d）
    #
    # 锚点：base_x_2d = -0.8（与 A/B/C 字母标签同一 x），
    # 使下端 pA_2d 在视觉上正好位于 A 标签旁边，清晰地从 A 延伸到 C。
    # 在箭头与线 B 的交叉点处加 tick 标记，消除"看起来只到 B"的歧义。
    # ================================================================
    base_x_2d = -0.8

    # 下端：线 A 上
    pA_2d = np.array([base_x_2d, line_y(base_x_2d, -MARGIN)])
    # 线 B 交叉点（用于 tick 标记）
    pB_2d = pA_2d + 1.0 * perp_dist * n_hat
    # 上端：线 C 上
    pC_2d = pA_2d + 2.0 * perp_dist * n_hat

    # === numpy 断言验证 ===
    assert abs(pA_2d[1] - line_y(pA_2d[0], -MARGIN)) < 1e-9, \
        f"2d 下端不在线 A: {pA_2d[1]:.9f} != {line_y(pA_2d[0],-MARGIN):.9f}"
    assert abs(pB_2d[1] - line_y(pB_2d[0], 0.0))    < 1e-9, \
        f"2d 中点不在线 B: {pB_2d[1]:.9f} != {line_y(pB_2d[0],0.0):.9f}"
    assert abs(pC_2d[1] - line_y(pC_2d[0], +MARGIN)) < 1e-9, \
        f"2d 上端不在线 C: {pC_2d[1]:.9f} != {line_y(pC_2d[0],+MARGIN):.9f}"

    # 画双向箭头（A → C，完整 2d）
    ax.annotate("", xy=tuple(pC_2d), xytext=tuple(pA_2d),
                arrowprops=dict(arrowstyle="<->", color=TEXT_COLOR,
                                linewidth=2.0, mutation_scale=15),
                zorder=6)

    # 在箭头与线 B 的交叉点处画横向 tick 标记（消除视觉歧义）
    tick_half_len = 0.18
    tick_lo = pB_2d - tick_half_len * tang
    tick_hi = pB_2d + tick_half_len * tang
    ax.plot([tick_lo[0], tick_hi[0]], [tick_lo[1], tick_hi[1]],
            color=TEXT_COLOR, lw=1.8, solid_capstyle="round", zorder=7)

    # "2d (margin)" 标签：放在箭头左侧，在线 B 与线 C 之间（箭头上半段）
    # 取 B 点到 C 点的中点，然后向 -tang 偏移
    mid_BC = (pB_2d + pC_2d) / 2.0        # B 与 C 之间的中点
    lab_2d = mid_BC - 1.35 * tang          # 向左偏移（-tang 方向）
    ax.text(lab_2d[0], lab_2d[1],
            r"$2d$ (margin)",
            color=TEXT_COLOR, fontsize=12, ha="center", va="center",
            fontweight="bold", zorder=7)

    # ================================================================
    # [2] d 双向箭头：线 A → 线 B（半 margin = d）
    # ================================================================
    base_x_d = 2.0

    pA_d_raw = np.array([base_x_d, line_y(base_x_d, -MARGIN)])
    pB_d_raw = pA_d_raw + 1.0 * perp_dist * n_hat

    assert abs(pA_d_raw[1] - line_y(pA_d_raw[0], -MARGIN)) < 1e-9, \
        f"d 下端不在线 A: {pA_d_raw[1]:.9f}"
    assert abs(pB_d_raw[1] - line_y(pB_d_raw[0], 0.0)) < 1e-9, \
        f"d 上端不在线 B: {pB_d_raw[1]:.9f}"

    # 沿切向偏移，避开支撑向量散点
    shift_d = +0.55 * tang
    pA_d = pA_d_raw + shift_d
    pB_d = pB_d_raw + shift_d

    ax.annotate("", xy=tuple(pB_d), xytext=tuple(pA_d),
                arrowprops=dict(arrowstyle="<->", color=TEXT_COLOR,
                                linewidth=1.5, mutation_scale=12),
                zorder=6)

    # "d" 标签：放在 d 箭头右侧中点
    mid_d = (pA_d + pB_d) / 2.0
    lab_d = mid_d + 0.55 * tang
    ax.text(lab_d[0], lab_d[1], r"$d$",
            color=TEXT_COLOR, fontsize=13, ha="center", va="center",
            zorder=7)

    # ------------------------------------------------------------------
    # 公式标签（精确贴靠各自对应线的右侧）
    # ------------------------------------------------------------------
    fx = 6.2
    y_on_C = line_y(fx, +MARGIN)
    y_on_B = line_y(fx,  0.0)
    y_on_A = line_y(fx, -MARGIN)

    seg_len_x = 0.6
    for y_on, ls, lw in [(y_on_C, "--", 2.2),
                          (y_on_B, "-",  3.2),
                          (y_on_A, "--", 2.2)]:
        ax.plot([fx - seg_len_x, fx - 0.05],
                [y_on + SLOPE * (-seg_len_x), y_on - 0.05 * SLOPE],
                color=LINE_COLOR, lw=lw, linestyle=ls, zorder=6)

    label_x_start = fx + 0.10
    ax.text(label_x_start, y_on_C,
            r"$w_d^{\,T}x + b_d = +1$",
            color=TEXT_COLOR, fontsize=12, ha="left", va="center", zorder=7)
    ax.text(label_x_start, y_on_B,
            r"$w^{\,T}x + b = 0$",
            color=TEXT_COLOR, fontsize=12, ha="left", va="center",
            fontweight="bold", zorder=7)
    ax.text(label_x_start, y_on_A,
            r"$w_d^{\,T}x + b_d = -1$",
            color=TEXT_COLOR, fontsize=12, ha="left", va="center", zorder=7)

    # ------------------------------------------------------------------
    # 坐标轴（箭头风格，无数字刻度）
    # ------------------------------------------------------------------
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    arrow_kw = dict(arrowstyle="-|>", color="#555555",
                    linewidth=1.8, mutation_scale=18)
    ax.annotate("", xy=(x_hi - 0.15, y_lo + 0.5),
                xytext=(x_lo + 0.3,  y_lo + 0.5),
                arrowprops=arrow_kw, zorder=1)
    ax.annotate("", xy=(x_lo + 0.3,  y_hi - 0.15),
                xytext=(x_lo + 0.3,  y_lo + 0.5),
                arrowprops=arrow_kw, zorder=1)

    ax.set_title("SVM 硬间隔：最大化 margin 与支撑向量", pad=10)
    fig.tight_layout()
    fig.savefig(OUT_PATH, format="png", dpi=200, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
