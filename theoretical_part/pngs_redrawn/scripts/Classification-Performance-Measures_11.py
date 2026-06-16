"""重绘: Classification-Performance-Measures_11

一条 ROC 曲线 (Receiver Operating Characteristic)。横轴 FPR (假正例率),
纵轴 TPR (真正例率 / 召回率)。曲线从 (0,0) 快速跃升、紧贴左上角后水平
延伸至 (1,1), 呈典型阶梯状, 表示一个表现良好的二分类器随判定阈值变化时
FPR 与 TPR 的关系。曲线越靠近左上角 (AUC 越大) 模型越优。

数据来源 (复用 notebook chp8 05-ROC):
    sklearn digits 手写数字, 构造有偏二分类 (digit==9 记为正类 1, 其余为 0),
    LogisticRegression 的 decision_function 得到判定分数, 再由
    sklearn.metrics.roc_curve 计算 (fprs, tprs)。random_state=666 复现高 AUC
    (约 0.982) 阶梯曲线。

原图: 蓝色折线 ROC, 无图例/无对角线/无标题。本重绘在统一风格下补 y=x 随机
基线虚线与 AUC 标注作为教学增强 (明确为原创增强, 非像素复制)。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Classification-Performance-Measures_11.png"


def compute_roc():
    """复用 notebook 流程: digits + 有偏标签 + 逻辑回归 decision_function。"""
    digits = load_digits()
    X = digits.data
    y = digits.target.copy()
    # 构造有偏二分类: digit==9 为正类 1, 其余为 0。
    y[digits.target == 9] = 1
    y[digits.target != 9] = 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=666
    )

    # max_iter=1000 以消除未收敛警告, 不影响图形。
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    decision_scores = clf.decision_function(X_test)

    fprs, tprs, _ = roc_curve(y_test, decision_scores)
    auc = roc_auc_score(y_test, decision_scores)
    return fprs, tprs, auc


def main():
    blue = _style.PALETTE[0]
    gray = _style.PALETTE[7]

    fprs, tprs, auc = compute_roc()

    fig, ax = _style.new_ax()

    # 随机猜测基线 (原创教学增强), y=x 对角虚线。
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        color=gray,
        alpha=0.8,
        label="随机猜测 (AUC = 0.5)",
    )

    # 主 ROC 曲线: 阶梯感来自 roc_curve 的离散点, 用 steps-post 还原。
    ax.plot(
        fprs,
        tprs,
        drawstyle="steps-post",
        color=blue,
        linewidth=2.2,
        label=f"ROC 曲线 (AUC = {auc:.3f})",
    )
    # 半透明填充凸显 AUC 面积。
    ax.fill_between(
        fprs,
        tprs,
        step="post",
        color=blue,
        alpha=0.12,
    )

    ax.set_xlabel("FPR (False Positive Rate)")
    ax.set_ylabel("TPR (True Positive Rate)")

    # 原图范围: matplotlib 默认留白 ~ -0.05 ~ 1.05。
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # "越靠近左上角越好" 教学箭头。
    ax.annotate(
        "越靠近左上角越好\n(面积 / AUC 越大)",
        xy=(0.03, 0.97),
        xytext=(0.40, 0.55),
        ha="left",
        va="center",
        fontsize=10,
        color="#333333",
        arrowprops=dict(
            arrowstyle="->",
            color="#555555",
            lw=1.4,
            connectionstyle="arc3,rad=-0.2",
        ),
    )

    ax.legend(loc="lower right")

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
