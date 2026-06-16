"""重绘: Classification-Performance-Measures_4

精准率 (precision) 与召回率 (recall) 随决策阈值 (threshold) 变化的双折线图。

数据来源: sklearn digits 手写数字数据集, 将 digit==9 设为正类 1、其余为 0,
构造类别不平衡的二分类问题; 用 LogisticRegression 训练后取 decision_function(X_test)
的决策分数, 再用 precision_recall_curve 生成 precisions / recalls / thresholds 三组
数组绘制 (非解析曲线, 而是由真实模型评分计算得到)。

蓝线 precisions 随阈值升高单调上升至 1.0, 橙线 recalls 随阈值升高单调下降至接近 0,
两条线在中部 (阈值约 -2、纵值约 0.85) 交叉, 直观展示 precision 与 recall 此消彼长的
权衡关系, 用于帮助选择决策阈值。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Classification-Performance-Measures_4.png"


def build_curves():
    """复现 notebook 的数据/模型流水线, 返回 thresholds / precisions / recalls。"""
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target.copy()
    # digit==9 设为正类 1, 其余为 0 -> 类别不平衡二分类。
    y[digits.target == 9] = 1
    y[digits.target != 9] = 0

    # random_state=666 保证与 notebook 可复现一致的划分。
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

    # max_iter 调大避免默认 solver 不收敛 (原 notebook 有 ConvergenceWarning),
    # 不影响曲线形态。
    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train, y_train)

    decision_scores = log_reg.decision_function(X_test)
    precisions, recalls, thresholds = precision_recall_curve(y_test, decision_scores)

    # precision_recall_curve 返回的 thresholds 比 precisions/recalls 少 1,
    # 绘图时 precisions/recalls 切片 [:-1] 对齐 (常见坑)。
    return thresholds, precisions[:-1], recalls[:-1]


def crossover_point(thresholds, precisions, recalls):
    """找两条曲线交叉处 (|precision - recall| 最小) 的阈值与纵值, 用于注释。"""
    diff = np.abs(precisions - recalls)
    idx = int(np.argmin(diff))
    x = thresholds[idx]
    y = 0.5 * (precisions[idx] + recalls[idx])
    return x, y


def main():
    thresholds, precisions, recalls = build_curves()

    fig, ax = _style.new_ax()

    # 沿用原图默认色环含义: precisions 蓝, recalls 橙 (统一调色板的 blue / amber)。
    c_prec = _style.PALETTE[0]  # blue
    c_rec = _style.PALETTE[3]  # amber

    ax.plot(thresholds, precisions, color=c_prec, linewidth=2.0,
            label="precisions (精准率)", zorder=3)
    ax.plot(thresholds, recalls, color=c_rec, linewidth=2.0,
            label="recalls (召回率)", zorder=3)

    # 标注交叉点: precision 与 recall 的平衡区, 强调 trade-off 含义。
    cx, cy = crossover_point(thresholds, precisions, recalls)
    ax.axvline(cx, color="#999999", linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)
    ax.scatter([cx], [cy], color="#333333", s=45, zorder=5)
    ax.annotate(
        f"平衡点\nthreshold≈{cx:.1f}",
        xy=(cx, cy),
        xytext=(cx + 6.0, cy - 0.18),
        fontsize=10,
        color="#333333",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
    )

    ax.set_xlabel("thresholds (决策阈值)")
    ax.set_ylabel("score")
    ax.set_title("Precision-Recall 随阈值的权衡")

    # 锁定与原图一致的观察窗口 (x ≈ -17~20): sklearn 不同版本在低精准率尾部
    # 会产生更负的决策分数, 这里裁剪视野使中段交叉区与原图框架对齐。
    ax.set_xlim(-18, 21)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15, 20])
    ax.legend(loc="lower left")

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
