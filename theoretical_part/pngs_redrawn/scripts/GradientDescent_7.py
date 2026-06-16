"""Redraw of GradientDescent_7: 2D gradient-descent on a loss surface.

z = x**2 + 2*y**2  (elliptic bowl, minimum at origin)

* Red elliptic contours (plt.contour + clabel).
* Gradient-descent trajectory from (-2.5, 2.5) to origin with a
  purple→orange gradient colouring via LineCollection.
* Direction arrows every few steps via ax.annotate.
* Start / end markers annotated in Chinese.

Styled with the unified _style.py (CJK-capable fonts, clean modern look).
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

import _style

OUT_PATH = (
    "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/GradientDescent_7.png"
)

CONTOUR_COLOR = _style.PALETTE[1]   # red   #EF4444
START_COLOR   = _style.PALETTE[4]   # violet
MIN_COLOR     = _style.PALETTE[2]   # green

# colormap for the trajectory: plasma goes violet→orange→yellow
TRAJ_CMAP = cm.plasma


def f(x, y):
    return x ** 2 + 2 * y ** 2


def grad_f(x, y):
    return np.array([2.0 * x, 4.0 * y])


def run_gd(x0, y0, eta=0.15, n_iters=50, epsilon=1e-9):
    """Return (N,2) array of visited (x, y) positions."""
    theta = np.array([x0, y0], dtype=float)
    history = [theta.copy()]
    for _ in range(n_iters):
        g = grad_f(theta[0], theta[1])
        theta = theta - eta * g
        history.append(theta.copy())
        if np.linalg.norm(g) < epsilon:
            break
    return np.array(history)


def main():
    # ------------------------------------------------------------------ grid
    xs = np.linspace(-3, 3, 400)
    ys = np.linspace(-3, 3, 400)
    xx, yy = np.meshgrid(xs, ys)
    zz = f(xx, yy)

    fig, ax = _style.new_ax(figsize=(6.2, 5.8))

    # ---------------------------------------------------------- contour plot
    levels = np.concatenate([
        np.arange(0.5, 3, 0.5),
        np.arange(3, 18, 1),
    ])
    cs = ax.contour(
        xx, yy, zz,
        levels=levels,
        colors=CONTOUR_COLOR,
        linewidths=0.85,
        alpha=0.80,
        zorder=1,
    )
    # label every other level to avoid clutter
    label_levels = levels[::2]
    ax.clabel(cs, levels=label_levels, inline=True, fontsize=7,
              fmt=lambda v: f"{v:.1f}" if v < 3 else f"{v:.0f}",
              inline_spacing=2)

    # ---------------------------------------------- gradient-descent path
    path = run_gd(-2.5, 2.5, eta=0.15, n_iters=50)
    N = len(path)

    # Build segments for LineCollection so we can colour each segment
    # individually along the colormap (0 = start = violet, 1 = end = yellow).
    segments = [
        [path[i], path[i + 1]]
        for i in range(N - 1)
    ]
    t = np.linspace(0, 1, N - 1)          # normalised step index
    lc = LineCollection(
        segments,
        cmap=TRAJ_CMAP,
        norm=plt.Normalize(0, 1),
        linewidth=2.2,
        zorder=3,
        label="梯度下降轨迹",
    )
    lc.set_array(t)
    ax.add_collection(lc)

    # Small dot markers at every point along the trajectory
    colors_pts = TRAJ_CMAP(np.linspace(0, 1, N))
    ax.scatter(
        path[:, 0], path[:, 1],
        c=colors_pts,
        s=18,
        zorder=4,
        edgecolors="white",
        linewidths=0.4,
    )

    # Direction arrows every k steps
    arrow_steps = list(range(0, min(N - 1, 14), 2))   # 0,2,4,...,12
    for i in arrow_steps:
        x0, y0_ = path[i]
        x1, y1_ = path[i + 1]
        col = TRAJ_CMAP(i / max(N - 2, 1))
        ax.annotate(
            "",
            xy=(x1, y1_),
            xytext=(x0, y0_),
            arrowprops=dict(
                arrowstyle="-|>",
                color=col,
                linewidth=0.0,   # shaft drawn by LineCollection
                mutation_scale=13,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=5,
        )

    # --------------------------------------------------------- start marker
    ax.scatter(
        [path[0, 0]], [path[0, 1]],
        s=90,
        color=START_COLOR,
        edgecolors="white",
        linewidths=1.5,
        zorder=6,
    )
    ax.annotate(
        "起始点",
        xy=(path[0, 0], path[0, 1]),
        xytext=(path[0, 0] - 0.15, path[0, 1] + 0.32),
        fontsize=10,
        color=START_COLOR,
        ha="right",
        va="bottom",
        zorder=7,
    )

    # ------------------------------------------------------ minimum marker
    ax.scatter(
        [0.0], [0.0],
        s=130,
        marker="*",
        color=MIN_COLOR,
        edgecolors="white",
        linewidths=1.2,
        zorder=6,
    )
    ax.annotate(
        "极小值\n(全局最优解)",
        xy=(0.0, 0.0),
        xytext=(0.60, -0.85),
        fontsize=9,
        color=MIN_COLOR,
        ha="left",
        va="top",
        arrowprops=dict(
            arrowstyle="-|>",
            color=MIN_COLOR,
            linewidth=1.3,
            mutation_scale=13,
            shrinkA=2,
            shrinkB=7,
        ),
        zorder=7,
    )

    # -------------------------------------------------------- legend proxies
    ax.plot([], [], color=START_COLOR,   linewidth=2.2,
            label=r"每步方向 $-\nabla f$ (下降最快)")
    ax.plot([], [], color=CONTOUR_COLOR, linewidth=0.85,
            label="等高线 (损失 J 取值，越外越大)")

    # ----------------------------------------------------------- colorbar
    sm = plt.cm.ScalarMappable(cmap=TRAJ_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("迭代进度（紫→橙）", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # -------------------------------------------------------- axes cosmetics
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$z = x^2 + 2y^2$")
    ax.grid(False)

    ax.legend(loc="upper right", framealpha=0.92, fontsize=8)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
