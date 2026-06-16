"""重绘: Logisitic-Regression_5

逻辑回归损失函数在真值 y=1 时的单分支曲线 cost = -log(p̂)。
横轴为模型预测概率 p̂ (0~1), 纵轴为该样本的损失 cost。
曲线随 p̂ 增大单调递减: p̂→0 时损失趋于正无穷, p̂=1 时损失为 0,
直观说明 "真值为 1 时 p̂ 越小则惩罚越大"。

纯函数生成, 无数据集。起点取 0.01 避开 log(0) 发散,
与原图顶端约 4.6 = -log(0.01) 一致。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Logisitic-Regression_5.png"


def main():
    # 起点取小正数 0.01 避开 log(0) 的正无穷, 采样足够密以保证左端急升段平滑。
    p = np.linspace(0.01, 1.0, 400)
    cost = -np.log(p)

    fig, ax = _style.new_ax()

    ax.plot(
        p,
        cost,
        color=_style.PALETTE[0],
        linewidth=2.0,
        label=r"真值 $y=1$ 时的损失: $-\log(\hat{p})$",
    )

    # 标注 p̂=1 时 cost=0 这一关键点 (预测正确, 损失为 0)。
    ax.scatter([1.0], [0.0], color=_style.PALETTE[1], s=45, zorder=5)
    ax.annotate(
        "预测正确, 损失为 0",
        xy=(1.0, 0.0),
        xytext=(0.52, 0.75),
        fontsize=10,
        color=_style.PALETTE[1],
        arrowprops=dict(arrowstyle="->", color=_style.PALETTE[1], lw=1.2),
    )

    # 标注 p̂→0 时 cost→+∞ (惩罚趋于无穷大)。
    ax.annotate(
        r"$\hat{p}\to 0$ 时 cost$\to+\infty$",
        xy=(0.02, 3.9),
        xytext=(0.22, 4.1),
        fontsize=10,
        color=_style.PALETTE[4],
        arrowprops=dict(arrowstyle="->", color=_style.PALETTE[4], lw=1.2),
    )

    ax.set_xlabel(r"预测概率 $\hat{p}$")
    ax.set_ylabel("损失 cost")
    ax.set_title(r"真值 $y=1$ 时的损失: $-\log(\hat{p})$")

    # 与原图比例对齐: x ~ 0~1, y ~ 0~4.6。
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.15, 4.8)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 1, 2, 3, 4])

    ax.legend(loc="upper right")

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
