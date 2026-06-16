"""重绘: Logisitic-Regression_3

绘制函数 y = -log(x) 在 x ∈ (0, 10] 区间内的单条曲线, 用于直观说明
逻辑回归损失函数推导中 -log() 的形状: 单调递减、过点 (1, 0)、左端急升趋正无穷,
为后续 y=1 时损失 -log(p̂) 的形态做铺垫。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Logisitic-Regression_3.png"


def main():
    # 起点取极小正数 1e-3 避开 log(0) 的负无穷，采样足够密以保证左端急升段平滑。
    x = np.linspace(1e-3, 10, 1000)
    y = -np.log(x)

    fig, ax = _style.new_ax(figsize=(6, 4))

    ax.plot(x, y, color=_style.PALETTE[0], linewidth=2.2)

    # 强调曲线过关键点 (1, 0)。
    ax.scatter([1.0], [0.0], color=_style.PALETTE[0], s=55, zorder=5)
    ax.annotate(
        r"$(1,\ 0)$",
        xy=(1.0, 0.0),
        xytext=(2.0, 0.9),
        fontsize=11,
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
    )

    # 轴范围与原图保持一致
    ax.set_xlim(-0.2, 10)
    ax.set_ylim(-2.8, 5.0)

    # 公式作为标题
    ax.set_title(r"$y = -\log(x)$", fontsize=14)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
