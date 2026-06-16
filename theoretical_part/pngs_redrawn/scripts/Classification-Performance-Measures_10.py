"""重绘: Classification-Performance-Measures_10 (系列汇总图)

一维 "score 轴" 分类示意图 (图 7-10 系列的最后一张, 汇总版)。12 个一维样本点
沿 score 横轴排列 (纵坐标固定在 y=0.02 处只为可见), 同时叠加三条不同位置的竖直
决策边界 / 阈值:
    - Decision Boundary-1: score = 0.025   (阈值最靠右)
    - Decision Boundary-2: score = 0.0
    - Decision Boundary-3: score = -0.019  (阈值最靠左)

概念: 阈值从右向左移动时, 被判为正类 (落在边界右侧) 的样本越来越多, TP 与 FP
同时增多 -> TPR 与 FPR 一起上升 (与 Precision-Recall 此消彼长相反)。具体计数:
    DB-1 (x= 0.025): TP=2, FP=0  -> TPR=2/6=0.33, FPR=0/6=0.00
    DB-2 (x= 0.0  ): TP=4, FP=1  -> TPR=4/6=0.67, FPR=1/6=0.16
    DB-3 (x=-0.019): TP=6, FP=2  -> TPR=6/6=1.00, FPR=2/6=0.33

坐标与 _7/_8/_9 共用同一组手工合成点, 复现时不可改动, 否则 FPR/TPR 数值对不上。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/"
    "Classification-Performance-Measures_10.png"
)

# 作者固定坐标 (合成手工值): 与 _7/_8/_9 完全一致, 复现时不可改动。
POINTS_0 = [-0.04, -0.035, -0.03, -0.02, -0.015, 0.01]  # 类别 0
POINTS_1 = [0.04, 0.03, 0.02, 0.008, -0.009, -0.018]    # 类别 1
Y_LEVEL = 0.02                                          # 所有点抬到的可见高度

# 三条决策边界 (x 位置, 名称, 颜色) —— 与单图保持配色一致。
BOUNDARIES = [
    (0.025, "Decision Boundary-1", _style.PALETTE[3]),   # amber, 最靠右
    (0.0, "Decision Boundary-2", _style.PALETTE[0]),     # blue
    (-0.019, "Decision Boundary-3", _style.PALETTE[2]),  # green, 最靠左
]


def main():
    p0 = np.asarray(POINTS_0)
    p1 = np.asarray(POINTS_1)

    # 计数自检, 保证与正文 TPR/FPR 结论一致。
    expected = {0.025: (2, 0), 0.0: (4, 1), -0.019: (6, 2)}
    for bx, (etp, efp) in expected.items():
        tp = int(np.sum(p1 > bx))
        fp = int(np.sum(p0 > bx))
        assert (tp, fp) == (etp, efp), f"边界 {bx}: TP/FP={tp}/{fp}, 期望 {etp}/{efp}"

    star_blue = _style.PALETTE[0]    # 类别 0 散点
    plus_orange = _style.PALETTE[5]  # 类别 1 散点 (cyan 系, 与三条边界配色区分)

    fig, ax = _style.new_ax(figsize=(7.5, 5.2))

    # --- 三条竖直决策边界线: 从 y=0 画到 y=0.5 ----------------------------
    for bx, name, color in BOUNDARIES:
        ax.plot(
            [bx, bx],
            [0.0, 0.5],
            color=color,
            linewidth=2.2,
            label=name,
            zorder=2,
        )

    # --- 两类散点 (星形=0, 加号=1), 同一水平高度 y=0.02 -------------------
    ax.scatter(
        p0,
        np.full(len(p0), Y_LEVEL),
        marker="*",
        s=170,
        color=star_blue,
        edgecolors="white",
        linewidths=0.5,
        zorder=5,
        label="0",
    )
    ax.scatter(
        p1,
        np.full(len(p1), Y_LEVEL),
        marker="+",
        s=170,
        linewidths=2.6,
        color=plus_orange,
        zorder=5,
        label="1",
    )

    # --- 阈值左移方向箭头 (原创增强, 强化"同向变化") ---------------------
    ax.annotate(
        "",
        xy=(-0.030, 0.40),
        xytext=(0.030, 0.40),
        arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.8),
    )
    ax.text(
        0.0,
        0.435,
        "阈值左移 → 判为正类的样本增多",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#333333",
    )
    ax.text(
        0.0,
        0.30,
        "TPR 与 FPR 同时上升\n(与 Precision-Recall 相反)",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#F5F7FA",
            edgecolor="#CCCCCC",
            lw=0.8,
        ),
    )

    # 给每条边界标注其 TPR/FPR, 呼应 _7/_8/_9。
    label_info = [
        (0.025, "TPR 0.33\nFPR 0.00"),
        (0.0, "TPR 0.67\nFPR 0.16"),
        (-0.019, "TPR 1.00\nFPR 0.33"),
    ]
    for bx, txt in label_info:
        ax.text(
            bx,
            0.515,
            txt,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#555555",
        )

    # --- 轴设置: x=score, y 仅占位无语义 --------------------------------
    ax.set_xlabel("score")
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.spines["left"].set_visible(False)

    ax.set_ylim(-0.02, 0.62)
    ax.set_xlim(-0.052, 0.052)

    ax.legend(loc="center right", framealpha=0.92)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
