"""重绘: Classification-Performance-Measures_12

在同一张 FPR-TPR 坐标系中比较两个模型的 ROC 曲线 (A 与 B):
    - 两条曲线都从原点 (0,0) 出发、到右上角 (1,1) 结束;
    - 两条都凹向左上角 (单调递增、上凸);
    - 曲线 B 整体压在 A 之上 (更靠近左上角 (0,1)), 即 B 的 AUC 更大,
      对应的模型更优。

这是一张示意性概念图 (纯函数曲线, 非真实数据):
    B: TPR = FPR ** 0.35  (更靠左上)
    A: TPR = FPR ** 0.60  (相对靠下)

风格还原原图的 "手绘感":
    - 坐标轴用带箭头样式, 只保留左轴和下轴 (去掉上/右边框);
    - 在原点、x 轴右端、y 轴顶端手动标注 0 / 1 / 1;
    - y=1 处一条横向点状参考线, x=1 处一条竖向点状参考线, 交于 (1,1);
    - A、B 文字标注放在各自曲线中上段, 蓝色 (贴近原图)。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "Classification-Performance-Measures_12.png"
)


def main():
    # --- 两条示意 ROC 曲线 (函数曲线, 凹向左上) ---------------------------
    fpr = np.linspace(0.0, 1.0, 400)
    tpr_B = fpr ** 0.35  # 上方曲线, AUC 更大 -> 模型更优
    tpr_A = fpr ** 0.60  # 下方曲线, AUC 更小

    # 自检: 两条都从 (0,0) 到 (1,1), 且 B 整体不低于 A。
    assert np.isclose(tpr_A[0], 0.0) and np.isclose(tpr_B[0], 0.0)
    assert np.isclose(tpr_A[-1], 1.0) and np.isclose(tpr_B[-1], 1.0)
    assert np.all(tpr_B + 1e-9 >= tpr_A), "B 曲线应整体压在 A 之上"

    fig, ax = _style.new_ax(figsize=(7, 5))
    # 概念示意图: 关闭统一风格的浅灰网格, 贴近原图干净背景。
    ax.grid(False)

    # --- 两条 ROC 曲线 (均为黑色) ----------------------------------------
    ax.plot(fpr, tpr_B, color="black", linewidth=2.0, zorder=3)
    ax.plot(fpr, tpr_A, color="black", linewidth=2.0, zorder=3)

    # --- y=1 与 x=1 的点状参考线, 交于 (1,1) ------------------------------
    ax.plot([0, 1], [1, 1], linestyle=":", color="#888888", linewidth=1.2, zorder=1)
    ax.plot([1, 1], [0, 1], linestyle=":", color="#888888", linewidth=1.2, zorder=1)

    # --- A / B 文字标注 (蓝色, 贴近原图), 放在各自曲线中上段 --------------
    blue = _style.PALETTE[0]
    ax.text(
        0.30, tpr_B[np.argmin(np.abs(fpr - 0.30))] + 0.045,
        "B", color=blue, fontsize=20, fontstyle="italic",
        ha="center", va="bottom", zorder=4,
    )
    ax.text(
        0.46, tpr_A[np.argmin(np.abs(fpr - 0.46))] - 0.085,
        "A", color=blue, fontsize=20, fontstyle="italic",
        ha="center", va="top", zorder=4,
    )

    # --- 带箭头的坐标轴样式: 只保留左轴和下轴, 末端加箭头 -----------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 左/下轴落在数据原点上。
    ax.spines["left"].set_position(("data", 0.0))
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # 轴范围: 留出箭头与标注空间。
    ax.set_xlim(0.0, 1.12)
    ax.set_ylim(0.0, 1.12)

    # 去掉默认刻度, 改为手动文字标注 (贴近原图手绘感)。
    ax.set_xticks([])
    ax.set_yticks([])

    # 箭头: 用 annotate 在轴末端画箭头头部。
    arrow = dict(arrowstyle="-|>", color="black", lw=1.4)
    ax.annotate("", xy=(1.12, 0.0), xytext=(0.0, 0.0), arrowprops=arrow, zorder=2)
    ax.annotate("", xy=(0.0, 1.12), xytext=(0.0, 0.0), arrowprops=arrow, zorder=2)

    # 手动刻度文字: 原点 0, x 轴右端 1, y 轴顶端 1。
    ax.text(-0.025, -0.03, "0", fontsize=15, ha="right", va="top")
    ax.text(1.0, -0.04, "1", fontsize=15, ha="center", va="top")
    ax.text(-0.04, 1.0, "1", fontsize=15, ha="right", va="center")

    # 轴标签: FPR (x), TPR (y), 放在原图对应位置。
    ax.text(0.5, -0.10, "FPR", fontsize=15, ha="center", va="top")
    ax.text(-0.13, 0.5, "TPR", fontsize=15, ha="center", va="center", rotation=90)

    ax.set_aspect("equal", adjustable="box")

    # 直接保存 (finalize 会触发 tight_layout, 但这里手动布局更可控)。
    fig.savefig(OUT_PATH, format="png", dpi=200, bbox_inches="tight")
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
