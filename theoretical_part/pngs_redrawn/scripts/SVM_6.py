"""Redraw of SVM_6: the SVM "maximum-margin decision boundary" concept diagram.

Two linearly separable classes of 2D points are drawn (red cluster in the
upper region = class +1, blue cluster in the lower region = class -1). A hard
margin linear SVM is fitted; from its weight vector w = coef_[0] and bias
b = intercept_[0] three parallel lines are drawn analytically:

    decision boundary :  w . x + b =  0   (solid)
    upper margin       :  w . x + b = +1   (dashed)
    lower margin       :  w . x + b = -1   (dashed)

The two dashed lines just graze the support vectors (2 red, 1 blue in this
case) and bound the empty "margin" corridor. Each support vector is connected
to the decision boundary by a short perpendicular segment (its foot = the
point projected along w) that marks the distance d -- the signature element of
the original sketch. A double-headed arrow annotates the full margin = 2d.

Faithful to the original screenshot: red upper cluster, blue lower cluster,
deep blue/violet lines (middle solid, two parallel margin lines), thin dark
perpendicular connectors from support vectors to the boundary, axis arrows with
no numeric ticks (hand-drawn schematic look). Styled with the unified _style.py.

Data: a synthetic, well-separated, linearly separable pair of 2D Gaussian
clusters via sklearn make_blobs with a fixed seed. Model: a hard-margin linear
SVM (sklearn.svm.SVC(kernel='linear', C=1e9)) so the margin lines hug the
support vectors and the corridor stays clean; SVC exposes support_vectors_
directly, which is more convenient than LinearSVC for this figure.
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_6.png"

RED = _style.PALETTE[1]    # red    #EF4444  -> class +1 (upper cluster)
BLUE = _style.PALETTE[0]   # blue   #2563EB  -> class -1 (lower cluster)
LINE_COLOR = "#3B2DD0"     # deep blue/violet for the three parallel lines
CONNECT_COLOR = "#1F2937"  # near-black thin perpendicular connectors
TEXT_COLOR = "#1F2937"


def main():
    rng = 6
    # Two well-separated clusters; small spread + pulled-apart centers so the
    # problem is comfortably linearly separable (hard margin won't fail).
    X, y = make_blobs(
        n_samples=14,
        centers=[(-1.5, -1.4), (1.5, 1.4)],
        cluster_std=0.62,
        random_state=rng,
    )
    # Class 0 -> lower-left blue (label -1) ; class 1 -> upper-right red (+1).

    # --- hard-margin linear SVM --------------------------------------------
    # SVC(kernel='linear', C=1e9) gives a clean hard margin and exposes
    # support_vectors_ directly.
    svc = SVC(kernel="linear", C=1e9)
    svc.fit(X, y)

    w = svc.coef_[0]              # (w0, w1)
    b = svc.intercept_[0]
    sv = svc.support_vectors_     # the support vectors (~2 red, 1 blue)

    fig, ax = _style.new_ax(figsize=(6.4, 4.8))

    # --- view box -----------------------------------------------------------
    x_lo, x_hi = -3.4, 3.4
    y_lo, y_hi = -3.2, 3.2
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # --- three parallel lines (analytic, exactly the reuse-hint formulas) ---
    # w0*x0 + w1*x1 + b = c  =>  x1 = -w0/w1 * x0 - b/w1 + c/w1
    plot_x = np.linspace(x_lo, x_hi, 300)
    base_y = -w[0] / w[1] * plot_x - b / w[1]
    mid_y = base_y                    # decision boundary  (c = 0)
    up_y = base_y + 1 / w[1]          # upper margin       (c = +1)
    down_y = base_y - 1 / w[1]        # lower margin       (c = -1)

    def clip(xv, yv):
        m = (yv >= y_lo) & (yv <= y_hi)
        return xv[m], yv[m]

    xm, ym = clip(plot_x, mid_y)
    xu, yu = clip(plot_x, up_y)
    xd, yd = clip(plot_x, down_y)

    # Decision boundary: solid; the two margin lines: dashed.
    ax.plot(xm, ym, color=LINE_COLOR, linewidth=2.6, solid_capstyle="round",
            zorder=2, label="决策边界  $w^Tx+b=0$")
    ax.plot(xu, yu, color=LINE_COLOR, linewidth=2.0, linestyle=(0, (6, 5)),
            zorder=2, label="上边界  $w^Tx+b=+1$")
    ax.plot(xd, yd, color=LINE_COLOR, linewidth=2.0, linestyle=(0, (6, 5)),
            zorder=2, label="下边界  $w^Tx+b=-1$")

    # --- scatter the two classes -------------------------------------------
    ax.scatter(
        X[y == 0, 0], X[y == 0, 1],
        color=BLUE, s=150, edgecolors="white", linewidths=1.3, zorder=3,
        label="类别 -1",
    )
    ax.scatter(
        X[y == 1, 0], X[y == 1, 1],
        color=RED, s=150, edgecolors="white", linewidths=1.3, zorder=3,
        label="类别 +1",
    )

    # --- highlight support vectors (larger, dark-edged) --------------------
    sv_colors = [RED if svc.predict([p])[0] == 1 else BLUE for p in sv]
    ax.scatter(
        sv[:, 0], sv[:, 1],
        s=320, facecolors="none", edgecolors=CONNECT_COLOR, linewidths=2.0,
        zorder=4,
    )

    # --- perpendicular connectors: support vector -> decision boundary -----
    # Foot of the perpendicular from a point p onto the boundary w.x + b = 0:
    #   foot = p - (w.p + b)/||w||^2 * w   (project along w).
    w_norm_sq = float(w @ w)
    d = 1.0 / np.sqrt(w_norm_sq)   # distance from a margin line to the boundary
    for p in sv:
        signed = (w @ p + b) / w_norm_sq
        foot = p - signed * w
        ax.plot([p[0], foot[0]], [p[1], foot[1]],
                color=CONNECT_COLOR, linewidth=1.4, zorder=2.5)

    # --- margin (= 2d) annotation along the w direction --------------------
    # Draw a double-headed arrow from the lower margin to the upper margin,
    # passing through the boundary, centered on a clear part of the corridor.
    w_hat = w / np.sqrt(w_norm_sq)             # unit normal (points to +1 side)
    # Anchor on the decision boundary at a chosen x; solve for y on the line.
    anchor_x = -0.55
    anchor_y = -w[0] / w[1] * anchor_x - b / w[1]
    anchor = np.array([anchor_x, anchor_y])
    p_up = anchor + d * w_hat                  # on the upper margin
    p_dn = anchor - d * w_hat                  # on the lower margin

    ax.annotate(
        "", xy=p_up, xytext=p_dn,
        arrowprops=dict(arrowstyle="<|-|>", color=TEXT_COLOR, linewidth=1.6,
                        mutation_scale=14, shrinkA=0, shrinkB=0),
        zorder=5,
    )
    mid_pt = anchor + 0.02 * w_hat
    # Offset the label sideways (along the boundary direction) so it stays out
    # of the arrow / lines.
    tdir = np.array([-w_hat[1], w_hat[0]])     # tangent to the boundary
    label_pt = mid_pt + 0.95 * tdir
    ax.text(label_pt[0], label_pt[1], "间隔 margin = 2d",
            color=TEXT_COLOR, fontsize=11, ha="center", va="center",
            rotation=np.degrees(np.arctan2(w_hat[1], w_hat[0])) - 90,
            rotation_mode="anchor", zorder=5)

    # Label a single "d" beside one support-vector connector.
    p0 = sv[0]
    signed0 = (w @ p0 + b) / w_norm_sq
    foot0 = p0 - signed0 * w
    midseg = (p0 + foot0) / 2.0
    ax.annotate("d", xy=midseg, xytext=midseg + 0.42 * tdir,
                color=TEXT_COLOR, fontsize=13, fontweight="bold",
                ha="center", va="center", zorder=5)

    # --- support-vector legend proxy ---------------------------------------
    from matplotlib.lines import Line2D
    sv_proxy = Line2D([0], [0], marker="o", color="none",
                      markerfacecolor="none", markeredgecolor=CONNECT_COLOR,
                      markeredgewidth=2.0, markersize=13, label="支撑向量")

    # --- hand-sketch axes: arrows for x and y, no numeric ticks ------------
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    arrow_kw = dict(arrowstyle="-|>", color="#333333", linewidth=1.5,
                    mutation_scale=18)
    ax.annotate("", xy=(x_hi, y_lo), xytext=(x_lo, y_lo), arrowprops=arrow_kw,
                zorder=1)
    ax.annotate("", xy=(x_lo, y_hi), xytext=(x_lo, y_lo), arrowprops=arrow_kw,
                zorder=1)

    ax.set_title("SVM：最大化两类最近样本间隔的决策边界")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(sv_proxy)
    labels.append("支撑向量")
    ax.legend(handles, labels, loc="lower right", fontsize=8.5,
              framealpha=0.92)

    fig.tight_layout()
    fig.savefig(OUT_PATH, format="png", dpi=200, bbox_inches="tight")
    print("saved:", OUT_PATH)
    print("n support vectors:", len(sv), "colors:", sv_colors)


if __name__ == "__main__":
    main()
