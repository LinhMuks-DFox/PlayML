"""Redraw of SVM_11: Hard Margin SVM — decision boundary dragged by outlier.

Concept: motivates Soft Margin SVM.
  - Red cluster (class +1): ALL strictly ABOVE upper margin dashed line.
  - Blue main cluster (class -1): ALL strictly BELOW lower margin dashed line.
  - One blue outlier (RIGHT side): circle centre EXACTLY ON lower margin line.

The Hard Margin SVM must separate ALL points including the outlier, so the
decision boundary is "dragged" rightward/downward.

Geometry (all coordinates explicitly computed with numpy, not eye-balled):

  Decision boundary:  y = SLOPE * x + INTERCEPT
  Upper margin:       y = SLOPE * x + INTERCEPT + MV   (+1 / red side)
  Lower margin:       y = SLOPE * x + INTERCEPT - MV   (-1 / blue side)

  SLOPE      = -0.25   (moderate negative slope — visually tilted, not diagonal)
  INTERCEPT  =  5.5    (vertically centred in plot)
  MV         =  1.8    (generous margin for clear visual separation)

  Axis limits: x in [0, 10], y in [0, 10]

  Line y-values at plot corners (all verified in [0,10]):
    Upper margin:  y(0)=7.3   y(10)=4.8   both in [0,10] ✓
    Decision BDY:  y(0)=5.5   y(10)=3.0   both in [0,10] ✓
    Lower margin:  y(0)=3.7   y(10)=1.2   both in [0,10] ✓

  Red points: ALL satisfy y > y_um(x) = -0.25x + 7.3
  Blue main:  ALL satisfy y < y_lm(x) = -0.25x + 3.7
  Outlier:    centre EXACTLY ON y_lm(x), AND below y_db(x)
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_11.png"

RED_COLOR  = _style.PALETTE[1]   # #EF4444
BLUE_COLOR = _style.PALETTE[0]   # #2563EB

X_MIN, X_MAX = 0.0, 10.0
Y_MIN, Y_MAX = 0.0, 10.0

# ── Boundary geometry ─────────────────────────────────────────────────────────
SLOPE     = -0.25   # moderate negative slope (not too steep, clearly tilted)
INTERCEPT =  5.5    # DB passes through (0, 5.5) — vertically centred
MV        =  1.8    # vertical offset: upper = INTERCEPT+MV, lower = INTERCEPT-MV

# Derived intercepts
INTERCEPT_UM = INTERCEPT + MV   # 7.3
INTERCEPT_LM = INTERCEPT - MV   # 3.7


def y_db(x):
    return SLOPE * x + INTERCEPT


def y_um(x):
    return SLOPE * x + INTERCEPT_UM


def y_lm(x):
    return SLOPE * x + INTERCEPT_LM


def line_endpoints(slope, b_int, xmin=X_MIN, xmax=X_MAX,
                   ymin=Y_MIN, ymax=Y_MAX):
    """Return (x0, y0, x1, y1) for y = slope*x + b_int clipped to the axis box.

    Checks all four edges, deduplicates corners (rounded to 10 dp), then
    returns the left-most and right-most intersection points.
    Raises ValueError if fewer than 2 unique intersections found.
    """
    pts = {}

    def add(x, y):
        key = round(float(x), 10)
        if key not in pts:
            pts[key] = (float(x), float(y))

    # Left edge (x = xmin)
    y_l = slope * xmin + b_int
    if ymin <= y_l <= ymax:
        add(xmin, y_l)

    # Right edge (x = xmax)
    y_r = slope * xmax + b_int
    if ymin <= y_r <= ymax:
        add(xmax, y_r)

    if abs(slope) > 1e-12:
        # Bottom edge (y = ymin)
        x_b = (ymin - b_int) / slope
        if xmin < x_b < xmax:   # strictly interior to avoid corner duplicates
            add(x_b, ymin)
        # Top edge (y = ymax)
        x_t = (ymax - b_int) / slope
        if xmin < x_t < xmax:
            add(x_t, ymax)

    sorted_pts = sorted(pts.values(), key=lambda p: p[0])
    if len(sorted_pts) < 2:
        raise ValueError(
            f"Line y={slope}x+{b_int} has only {len(sorted_pts)} intersection(s) "
            f"with box [{xmin},{xmax}]x[{ymin},{ymax}]"
        )

    p0, p1 = sorted_pts[0], sorted_pts[-1]
    print(f"  Line(b={b_int:+.1f}): "
          f"({p0[0]:.4f},{p0[1]:.4f}) -> ({p1[0]:.4f},{p1[1]:.4f})  "
          f"[{len(sorted_pts)} endpoint(s) after dedup]")
    return p0[0], p0[1], p1[0], p1[1]


def build_dataset():
    """Hand-placed 2D points satisfying all Hard Margin SVM constraints.

    All points verified against:
      Red:     y > y_um(x) = -0.25x + 7.3    (strictly above upper margin)
      Blue:    y < y_lm(x) = -0.25x + 3.7    (strictly below lower margin)
      Outlier: y == y_lm(x_out)               (exactly on lower margin)
               y < y_db(x_out)               (below decision boundary)

    Verification table (computed with numpy):
      y_um(x) = -0.25*x + 7.3
      Red points — must satisfy ry > y_um(rx):
        (1.5,  9.2): y_um=6.925  9.2 > 6.925  gap=2.275  ✓
        (2.5,  9.5): y_um=6.675  9.5 > 6.675  gap=2.825  ✓
        (3.5,  9.2): y_um=6.425  9.2 > 6.425  gap=2.775  ✓
        (4.5,  9.5): y_um=6.175  9.5 > 6.175  gap=3.325  ✓
        (5.5,  9.0): y_um=5.925  9.0 > 5.925  gap=3.075  ✓
        (6.5,  9.3): y_um=5.675  9.3 > 5.675  gap=3.625  ✓
        (7.5,  8.8): y_um=5.425  8.8 > 5.425  gap=3.375  ✓
        (8.5,  9.1): y_um=5.175  9.1 > 5.175  gap=3.925  ✓

      y_lm(x) = -0.25*x + 3.7
      Blue main points — must satisfy by < y_lm(bx):
        (0.5,  2.5): y_lm=3.575  2.5 < 3.575  gap=1.075  ✓
        (1.5,  1.8): y_lm=3.325  1.8 < 3.325  gap=1.525  ✓
        (2.0,  2.6): y_lm=3.200  2.6 < 3.200  gap=0.600  ✓
        (3.0,  1.5): y_lm=2.950  1.5 < 2.950  gap=1.450  ✓
        (3.8,  2.0): y_lm=2.750  2.0 < 2.750  gap=0.750  ✓
        (4.8,  1.2): y_lm=2.500  1.2 < 2.500  gap=1.300  ✓

      Outlier at x_out=7.0:
        y_lm(7.0) = -0.25*7.0 + 3.7 = 1.95   (exactly on lower margin)
        y_db(7.0) = -0.25*7.0 + 5.5 = 3.75   (above outlier → outlier below DB) ✓
    """

    # Red cluster: strictly ABOVE upper margin  y_um(x) = -0.25x + 7.3
    red = np.array([
        [1.5,  9.2],
        [2.5,  9.5],
        [3.5,  9.2],
        [4.5,  9.5],
        [5.5,  9.0],
        [6.5,  9.3],
        [7.5,  8.8],
        [8.5,  9.1],
    ])

    # Blue main cluster: strictly BELOW lower margin  y_lm(x) = -0.25x + 3.7
    blue = np.array([
        [0.5,  2.5],
        [1.5,  1.8],
        [2.0,  2.6],
        [3.0,  1.5],
        [3.8,  2.0],
        [4.8,  1.2],
    ])

    # Blue outlier: centre EXACTLY ON lower margin at x_out = 7.0
    # y_lm(7.0) = -0.25*7.0 + 3.7 = 1.95  (computed with numpy)
    x_out = np.float64(7.0)
    y_out = np.float64(SLOPE) * x_out + np.float64(INTERCEPT_LM)  # exact: 1.95
    outlier = np.array([x_out, y_out])

    return red, blue, outlier


def verify_constraints(red, blue, outlier):
    """Assert all constraints; print summary. Raises if any fail."""
    print("\nVerifying geometric constraints...")
    ok = True

    for i, (rx, ry) in enumerate(red):
        um = y_um(rx)
        s = "OK" if ry > um else "FAIL"
        if s == "FAIL":
            ok = False
        print(f"  {s}  红[{i}] ({rx:.1f},{ry:.1f})  y_um={um:.3f}  gap={ry-um:.3f}")

    for i, (bx, by) in enumerate(blue):
        lm = y_lm(bx)
        s = "OK" if by < lm else "FAIL"
        if s == "FAIL":
            ok = False
        print(f"  {s}  蓝[{i}] ({bx:.1f},{by:.1f})  y_lm={lm:.3f}  gap={lm-by:.3f}")

    ox, oy = outlier
    lm_out = y_lm(ox)
    db_out = y_db(ox)
    diff   = abs(oy - lm_out)
    s1 = "OK" if diff < 1e-9 else "FAIL"
    s2 = "OK" if oy < db_out else "FAIL"
    if s1 == "FAIL" or s2 == "FAIL":
        ok = False
    print(f"  {s1}  离群点 ({ox:.1f},{oy:.6f})  y_lm={lm_out:.6f}  diff={diff:.2e}")
    print(f"  {s2}  离群点在决策边界下方: y={oy:.4f} < y_db={db_out:.4f}")

    # Extra: confirm all red strictly above their respective y_um values
    for i, (rx, ry) in enumerate(red):
        assert ry > y_um(rx), f"Red point [{i}] NOT above upper margin!"

    # Extra: confirm all blue strictly below their respective y_lm values
    for i, (bx, by) in enumerate(blue):
        assert by < y_lm(bx), f"Blue point [{i}] NOT below lower margin!"

    if not ok:
        raise AssertionError("Constraint verification FAILED!")
    print("All constraints passed. ✓\n")


def main():
    red, blue, outlier = build_dataset()
    verify_constraints(red, blue, outlier)

    # ── Clip lines to axis box ────────────────────────────────────────────────
    print("Line endpoint clipping:")
    umx0, umy0, umx1, umy1 = line_endpoints(SLOPE, INTERCEPT_UM)
    dbx0, dby0, dbx1, dby1 = line_endpoints(SLOPE, INTERCEPT)
    lmx0, lmy0, lmx1, lmy1 = line_endpoints(SLOPE, INTERCEPT_LM)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = _style.new_ax(figsize=(7, 5.5))
    ax.grid(False)

    # ── THREE lines: exactly ONE ax.plot call each ────────────────────────────
    #
    # 1. Upper margin — gray dashed
    ax.plot([umx0, umx1], [umy0, umy1],
            color="#999999", linewidth=2.2, linestyle=(0, (8, 5)),
            zorder=2, solid_capstyle="butt", dash_capstyle="butt")

    # 2. Decision boundary — solid black
    ax.plot([dbx0, dbx1], [dby0, dby1],
            color="#111111", linewidth=3.5, linestyle="-",
            zorder=3, antialiased=True,
            solid_capstyle="butt")

    # 3. Lower margin — gray dashed (same style as upper)
    ax.plot([lmx0, lmx1], [lmy0, lmy1],
            color="#999999", linewidth=2.2, linestyle=(0, (8, 5)),
            zorder=2, solid_capstyle="butt", dash_capstyle="butt")

    # ── Scatter points ────────────────────────────────────────────────────────

    # Red cluster: strictly above upper margin
    ax.scatter(
        red[:, 0], red[:, 1],
        s=180, c=RED_COLOR, marker="o",
        edgecolors="white", linewidths=1.2, zorder=5,
    )

    # Blue main cluster: strictly below lower margin
    ax.scatter(
        blue[:, 0], blue[:, 1],
        s=180, c=BLUE_COLOR, marker="o",
        edgecolors="white", linewidths=1.2, zorder=5,
    )

    # Blue outlier: centre EXACTLY ON lower margin — black ring = support vector
    # x_out=7.0, y_out=1.95 (computed by numpy above)
    ax.scatter(
        outlier[0], outlier[1],
        s=260, c=BLUE_COLOR, marker="o",
        edgecolors="#111111", linewidths=2.5, zorder=6,
    )

    # ── "Decision Boundary" label ─────────────────────────────────────────────
    # Arrow tip: ON the DB line at x = 1.5
    #   y_db(1.5) = -0.25*1.5 + 5.5 = 5.125
    # Text anchor: slightly right and above tip, in the margin zone
    #   Midpoint between UM and DB at x=3: y_mid = (y_um(3)+y_db(3))/2
    #   y_um(3)=6.55, y_db(3)=4.75 → midpoint=5.65
    ann_xp = np.float64(1.5)
    ann_yp = y_db(ann_xp)   # 5.125 — on the DB

    ann_xt = 3.0
    ann_yt = float(y_um(3.0) + y_db(3.0)) / 2.0   # 5.65 — midpoint of margin zone

    ax.annotate(
        "Decision Boundary",
        xy=(float(ann_xp), float(ann_yp)),
        xytext=(ann_xt, ann_yt),
        color=RED_COLOR,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        arrowprops=dict(
            arrowstyle="-|>",
            color=RED_COLOR,
            lw=1.2,
            mutation_scale=12,
        ),
        zorder=7,
    )

    # ── Axis limits and ticks ─────────────────────────────────────────────────
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks([])
    ax.set_yticks([])

    # Hide all four default spines; replace with arrow-tipped axes
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # Arrow-tipped x-axis
    ax.annotate("", xy=(X_MAX, Y_MIN), xytext=(X_MIN, Y_MIN),
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color="#333333",
                                linewidth=1.8, mutation_scale=22),
                zorder=1)
    # Arrow-tipped y-axis
    ax.annotate("", xy=(X_MIN, Y_MAX), xytext=(X_MIN, Y_MIN),
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color="#333333",
                                linewidth=1.8, mutation_scale=22),
                zorder=1)

    _style.finalize(fig, OUT_PATH)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
