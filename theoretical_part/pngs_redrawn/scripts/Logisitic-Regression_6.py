"""重绘: Logisitic-Regression_6

逻辑回归损失函数中 y=0 分支的代价曲线。横轴为 sigmoid 预测的概率 p̂ (0~1),
纵轴为代价 cost = -log(1 - p̂)。曲线从 p̂=0 处的 0 出发, 随 p̂ 增大缓慢上升,
当 p̂→1 时陡峭趋向正无穷, 直观说明: 当样本真值 y=0 时, 预测概率 p̂ 越大
(越偏向判为 1), 代价越大。

注意: 笔记正文里出现的 -log(p̂ - 1) 在 [0,1] 上无意义, 原图实际绘制的是
-log(1 - p̂), 即二元交叉熵 y=0 项的正确形式, 这里采用后者。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Logisitic-Regression_6.png"


def cost(p):
    """y=0 时的损失: -log(1 - p̂)。"""
    return -np.log(1.0 - p)


def main():
    # 定义域避开端点: linspace(0.01, 0.99, 200) 防止 log(0) 溢出。
    # 校验: -log(1 - 0.99) ≈ 4.6, 与原图右端纵值吻合。
    p = np.linspace(0.01, 0.99, 200)
    y = cost(p)

    fig, ax = _style.new_ax()

    ax.plot(p, y, color=_style.PALETTE[0], linewidth=2.0, label=r"$-\log(1 - \hat{p})$")

    # 右上角注释箭头: 点明 p̂→1 时 cost→+∞。
    ax.annotate(
        r"$\hat{p} \to 1$ 时 $\mathrm{cost} \to +\infty$",
        xy=(0.985, cost(0.985)),
        xytext=(0.50, 4.0),
        fontsize=10.5,
        color=_style.PALETTE[1],
        ha="center",
        arrowprops=dict(arrowstyle="->", color=_style.PALETTE[1], lw=1.3),
    )

    ax.set_xlabel(r"预测概率 $\hat{p}$")
    ax.set_ylabel(r"代价 cost $= -\log(1 - \hat{p})$")
    ax.set_title(r"$y=0$ 时的损失函数: $-\log(1 - \hat{p})$")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.2, 4.8)
    ax.legend(loc="upper left")

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
