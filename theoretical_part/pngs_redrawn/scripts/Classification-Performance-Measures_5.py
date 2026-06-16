"""重绘: Classification-Performance-Measures_5

精准率-召回率 (Precision-Recall) 曲线。横轴为 Precision, 纵轴为 Recall。
曲线整体从左上 (高召回、低精准) 缓慢下降, 在 Precision 接近 1.0 附近出现
急剧陡降, 直至 Recall=0。曲线急剧下降的拐点通常就是精准率与召回率达到
良好平衡的点。

数据来源: sklearn load_digits 数据集做 "数字9 vs 非9" 二分类,
LogisticRegression 取 decision_function 分数, 再用
sklearn.metrics.precision_recall_curve 得到 (precisions, recalls)。
对应原笔记最后一个绘图 cell id=d37e4cda (sklearn 阶梯+陡降版本)。

复现要点:
- train_test_split 固定 random_state=666 (与原笔记一致)。
- LogisticRegression 加 max_iter=10000 以稳定收敛并消除 ConvergenceWarning,
  曲线形状基本不变。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Classification-Performance-Measures_5.png"


def main():
    # 数据准备: load_digits, "数字9 vs 非9" 二分类。
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target.copy()
    y[digits.target == 9] = 1
    y[digits.target != 9] = 0

    # 固定 random_state=666, 与原笔记一致, 保证可复现。
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    # max_iter=10000 稳定收敛, 消除 ConvergenceWarning, 曲线形状不变。
    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train, y_train)
    decision_scores = log_reg.decision_function(X_test)

    # sklearn 版本: 阶梯状, 末端在 precision≈1.0 处陡降到 recall=0。
    precisions, recalls, _ = precision_recall_curve(y_test, decision_scores)

    fig, ax = _style.new_ax()

    ax.plot(
        precisions,
        recalls,
        color=_style.PALETTE[0],
        linewidth=2.0,
    )

    ax.set_xlabel("Precisions")
    ax.set_ylabel("Recalls")
    ax.set_title("精准率-召回率 (Precision-Recall) 曲线")

    # 与原图坐标范围对齐: x 约 0.3~1.0, y 为 0.0~1.0。
    ax.set_xlim(0.28, 1.02)
    ax.set_ylim(-0.03, 1.03)

    # 在 precision 接近 1.0 处的急剧下降拐点 ── 精准率/召回率的良好平衡点。
    # 取陡降段中部作为标注锚点 (precision≈0.96, recall≈0.7 附近)。
    knee_p = precisions[recalls <= 0.7][0] if (recalls <= 0.7).any() else 0.96
    knee_r = recalls[precisions >= knee_p][0]
    ax.annotate(
        "陡降拐点\n精准率/召回率平衡点",
        xy=(knee_p, knee_r),
        xytext=(0.45, 0.45),
        fontsize=10,
        color=_style.PALETTE[1],
        ha="center",
        arrowprops=dict(arrowstyle="->", color=_style.PALETTE[1], lw=1.2),
    )

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
