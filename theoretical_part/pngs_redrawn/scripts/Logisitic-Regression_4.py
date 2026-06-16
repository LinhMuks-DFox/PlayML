"""重绘: Logisitic-Regression_4

同屏对比绘制 y = log(x) 与 y = -log(x) 两条曲线, x ∈ (0, 10]。
蓝色递增曲线 log(x) 与橙色递减曲线 -log(x) 关于 x 轴互为镜像, 在 (1, 0) 处相交,
直观展示 -log(x) 是把 log(x) 上下翻转, 为后续构造逻辑回归代价函数 -log(p) 做铺垫。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Logisitic-Regression_4.png"


def main():
    # 起点取小正数 0.01 避开 log(0) 的奇点; log(0.01)≈-4.6, 还原原图左端 ±4.6 的视觉范围。
    # 采样足够密 (1000 点), 保证 x→0 的陡峭段平滑无折线感。
    x = np.linspace(0.01, 10, 1000)
    y_log = np.log(x)
    y_neg = -np.log(x)

    fig, ax = _style.new_ax()

    # y=0 淡色水平参考线, 强调两曲线过零点并关于 x 轴镜像。
    ax.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)

    ax.plot(x, y_log, color=_style.PALETTE[0], linewidth=2.0, label=r"$\log(x)$", zorder=3)
    ax.plot(x, y_neg, color=_style.PALETTE[3], linewidth=2.0, label=r"$-\log(x)$", zorder=3)

    # 标出两曲线交点 (1, 0)。
    ax.scatter([1.0], [0.0], color="#333333", s=45, zorder=5)
    ax.annotate(
        "(1, 0)",
        xy=(1.0, 0.0),
        xytext=(2.4, 1.4),
        fontsize=11,
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$\log(x)$ 与 $-\log(x)$ 关于 x 轴镜像")

    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-4.8, 4.8)
    ax.legend(loc="upper right")

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
