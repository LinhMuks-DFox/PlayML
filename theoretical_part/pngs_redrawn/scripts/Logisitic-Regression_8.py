"""重绘: Logisitic-Regression_8

逻辑回归的非线性 (圆形) 决策边界示意图。

概念:
  * 红色散点 (正类, 类别 1) 落在以原点为中心的圆内;
  * 蓝色散点 (负类, 类别 0) 落在圆外;
  * 黑色圆作为决策边界 (无填充, 仅描边), 对应多项式特征
    x1^2 + x2^2 - r^2 = 0。

这说明一条直线无法分开这种数据, 必须引入多项式特征 (x1^2 + x2^2)
才能用线性模型刻画圆形边界。本图是纯概念示意图, 直接用解析方程画圆,
不需要真的拟合 LogisticRegression。

数据生成沿用原 notebook 思路:
  np.random.seed(666); X = np.random.normal(0,1,(200,2)); y = (x1^2+x2^2<1.5)
原图坐标轴呈十字形 (spine 居中穿过原点), 用箭头表示坐标轴方向;
并去掉原截图右下角的 'www.imooc.co' 水印。
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Logisitic-Regression_8.png"

INSIDE_COLOR = _style.PALETTE[1]   # red  #EF4444 -> 圆内, 类别 1 (正类)
OUTSIDE_COLOR = _style.PALETTE[0]  # blue #2563EB -> 圆外, 类别 0 (负类)

# 阈值 1.5 (到原点距离平方), 边界半径 r = sqrt(1.5) ~= 1.225,
# 落在规格建议的 1.2~1.4 区间内。
THRESHOLD = 1.5
RADIUS = float(np.sqrt(THRESHOLD))


def make_data():
    """生成二维高斯随机点, 按到原点的距离平方分为圆内/圆外两类。

    为了让示意图更干净 (接近原截图: 约 8 个红点在圆内、约 20 个蓝点在
    圆外), 在边界附近留出一条间隙带, 剔除恰好压在决策边界上的点。
    """
    rng = np.random.RandomState(666)
    X = rng.normal(0.0, 1.0, (200, 2))
    r2 = X[:, 0] ** 2 + X[:, 1] ** 2

    # 留出边界附近的视觉间隙 (gap band): 内圈用 0.85*阈值, 外圈用 1.25*阈值。
    inside = r2 < THRESHOLD * 0.85
    outside = r2 > THRESHOLD * 1.25

    Xi = X[inside]
    Xo = X[outside]

    # 控制数量, 让红点稀疏、蓝点偏多, 贴近原图密度。
    Xi = Xi[:9]
    Xo = Xo[:22]
    return Xi, Xo


def main():
    Xi, Xo = make_data()

    fig, ax = _style.new_ax(figsize=(6.4, 5.0))

    # 散点: 圆内红 (正类), 圆外蓝 (负类)。白色描边让点更分明。
    ax.scatter(
        Xo[:, 0],
        Xo[:, 1],
        s=180,
        c=OUTSIDE_COLOR,
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
        label="圆外: 类别 0 (负类)",
    )
    ax.scatter(
        Xi[:, 0],
        Xi[:, 1],
        s=180,
        c=INSIDE_COLOR,
        edgecolors="white",
        linewidths=1.2,
        zorder=4,
        label="圆内: 类别 1 (正类)",
    )

    # 黑色圆: 决策边界, 无填充仅描边 (用参数方程画, 配合 aspect equal)。
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(
        RADIUS * np.cos(theta),
        RADIUS * np.sin(theta),
        color="black",
        linewidth=1.8,
        zorder=2,
        label="决策边界 (圆形)",
    )

    # 决策边界方程标注, 放在圆的右上外侧。
    ax.annotate(
        r"$x_1^2 + x_2^2 - r^2 = 0$",
        xy=(RADIUS * np.cos(np.pi / 4), RADIUS * np.sin(np.pi / 4)),
        xytext=(1.9, 2.5),
        fontsize=11,
        color="black",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
    )

    # 等比例坐标, 圆不变形。
    ax.set_aspect("equal", adjustable="box")

    lim = 3.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # --- 居中坐标轴 (十字形, 穿过原点), 带箭头, 复现原示意图风格 -----------
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    # 不要刻度数字与刻度线 (原图为纯示意图, 无坐标读数)。
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # 用箭头表示坐标轴方向 (x 向右, y 向上)。
    ax.plot(1, 0, ">", transform=ax.get_yaxis_transform(), clip_on=False,
            color="#333333", markersize=8, zorder=5)
    ax.plot(0, 1, "^", transform=ax.get_xaxis_transform(), clip_on=False,
            color="#333333", markersize=8, zorder=5)

    # 坐标轴标签贴在箭头附近。
    ax.text(lim, -0.18, r"$x_1$", fontsize=12, ha="right", va="top", color="#333333")
    ax.text(0.15, lim, r"$x_2$", fontsize=12, ha="left", va="top", color="#333333")

    ax.set_title("逻辑回归的非线性 (圆形) 决策边界")
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.9)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
