"""Redraw of GradientDescent_8: SGD search path on parameter-space contour map.

Fix v3:
  - Contour: strictly flat 2D top-down orthographic concentric circles.
    Uses a circular loss function (t1^2 + t2^2) so contours are perfect
    circles, matching the original reference image. A circular Patch clip
    ensures the outermost ring has a clean circular boundary with white
    background outside — no 3D rim, no bevel, no perspective distortion.
  - Color mapping: dark outer ring (near black) to light inner (near white),
    exactly matching the original reference image color order.
  - SGD path: zigzag convergence from upper-left toward the center, ending
    in the innermost bright region, illustrating SGD reaching the minimum.
"""

import sys
sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/GradientDescent_8.png"

PATH_COLOR  = "#5B4EA8"   # purple dots
LINK_COLOR  = "#2D3561"   # dark navy lines
ARROW_COLOR = "#1A1A2E"   # axis arrow color
TITLE_COLOR = "#B11E3A"   # red title


def J(t1, t2):
    """Circular loss surface: concentric circles centered at origin."""
    return t1 ** 2 + t2 ** 2


def make_sgd_path():
    """Hand-craft a zigzag convergence path ending inside the innermost ring."""
    np.random.seed(7)

    # Start upper-left, end near origin
    start = np.array([-1.2, 1.4])
    end   = np.array([0.05, 0.0])

    direction = end - start
    d_norm    = np.linalg.norm(direction)
    perp      = np.array([-direction[1], direction[0]]) / d_norm

    n = 14
    fracs = np.linspace(0.0, 1.0, n + 2)[1:-1] ** 0.75

    pts  = [start.copy()]
    sign = 1.0
    for f in fracs:
        base      = start + f * direction
        remaining = 1.0 - f
        amp       = 0.30 * remaining * d_norm + 0.03
        jitter    = np.random.uniform(0.85, 1.15)
        pt        = base + sign * amp * jitter * perp
        pt       += np.random.normal(0, 0.02, 2)
        # Keep inside outer boundary
        r = np.linalg.norm(pt)
        if r > 2.4:
            pt = pt / r * 2.4
        pts.append(pt.copy())
        sign *= -1.0

    pts.append(np.array([0.05, 0.0]))
    return np.array(pts)


def draw_arrow_axes(ax, x0, y0, x1, y1):
    """Draw theta_1 (horizontal) and theta_2 (vertical) arrows."""
    kw = dict(
        arrowstyle="-|>",
        color=ARROW_COLOR,
        lw=2.0,
        mutation_scale=20,
        shrinkA=0,
        shrinkB=0,
    )
    ax.annotate("", xy=(x1, y0), xytext=(x0, y0),
                arrowprops=kw, annotation_clip=False, zorder=10)
    ax.annotate("", xy=(x0, y1), xytext=(x0, y0),
                arrowprops=kw, annotation_clip=False, zorder=10)

    ax.text(x1 + 0.20, y0, r"$\theta_1$",
            fontsize=18, fontweight="bold", color=ARROW_COLOR,
            ha="left", va="center", zorder=10, clip_on=False)

    ax.text(x0, y1 + 0.18, r"$\theta_2$",
            fontsize=18, fontweight="bold", color=ARROW_COLOR,
            ha="center", va="bottom", zorder=10, clip_on=False)


def main():
    fig, ax = _style.new_ax(figsize=(6.8, 6.4))

    # ── loss surface grid ──────────────────────────────────────────────────────
    lim = 2.6
    g   = np.linspace(-lim, lim, 1000)
    T1, T2 = np.meshgrid(g, g)
    Z = J(T1, T2)

    # ── contour levels: 6 bands, equally spaced in radius so rings are uniform──
    # Outermost circle radius we want to show
    R_max  = 2.4
    n_bands = 6
    # Level edges at r^2: 0, (R/n)^2, (2R/n)^2, ...
    radii  = np.linspace(0.0, R_max, n_bands + 1)
    levels = radii ** 2   # z = r^2 for circular J

    # Colors: dark outer → light inner (matching original reference)
    # index 0 = innermost band, index n-1 = outermost band
    gray_colors = [
        "#F2F2F2",  # innermost  (near-white)
        "#D0D0D0",
        "#ABABAB",
        "#808080",
        "#555555",
        "#282828",  # outermost  (near-black)
    ]
    cmap_flat = ListedColormap(gray_colors)
    norm_flat = BoundaryNorm(levels, cmap_flat.N)

    # Draw filled contour — flat, no antialiasing banding artifacts
    cf = ax.contourf(
        T1, T2, Z,
        levels=levels,
        cmap=cmap_flat,
        norm=norm_flat,
        zorder=0,
        antialiased=True,
        extend="neither",
    )

    # White separator lines between bands
    ax.contour(
        T1, T2, Z,
        levels=levels[1:-1],
        colors="white",
        linewidths=1.4,
        zorder=1,
        alpha=0.9,
    )

    # ── clip everything to a clean circle — eliminates any rectangular or
    #    elliptical boundary that could look like a 3D rim ─────────────────────
    # In matplotlib >= 3.8 contourf returns a ContourSet whose artists are
    # accessed via cf.get_paths() or by iterating the axes' collections.
    # The safest cross-version approach: clip each PathCollection on the axes.
    circle_clip = mpatches.Circle(
        (0, 0), radius=R_max,
        transform=ax.transData,
        facecolor="none",
        edgecolor="none",
    )
    ax.add_patch(circle_clip)
    # Clip every PolyCollection / PathCollection that was added by contourf
    for artist in ax.get_children():
        if hasattr(artist, "set_clip_path"):
            artist.set_clip_path(circle_clip)

    # Draw a thin dark border on the circle boundary
    theta_c = np.linspace(0, 2 * np.pi, 500)
    ax.plot(
        R_max * np.cos(theta_c),
        R_max * np.sin(theta_c),
        color="#333333", lw=1.2, zorder=2,
    )

    # ── SGD path ───────────────────────────────────────────────────────────────
    path = make_sgd_path()
    px, py = path[:, 0], path[:, 1]

    ax.plot(px, py, color=LINK_COLOR, lw=1.6, zorder=3,
            solid_capstyle="round", solid_joinstyle="round")

    ax.scatter(px, py, s=80, c=PATH_COLOR, marker="o",
               edgecolors="white", linewidths=0.8, zorder=5)

    # Arrow on the last segment to show direction of convergence
    ax.annotate(
        "",
        xy=(px[-1], py[-1]),
        xytext=(px[-2], py[-2]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=LINK_COLOR,
            lw=1.8,
            mutation_scale=16,
            shrinkA=5,
            shrinkB=4,
        ),
        zorder=6,
    )

    # ── axes frame ─────────────────────────────────────────────────────────────
    ax.set_xlim(-lim - 0.9, lim + 1.0)
    ax.set_ylim(-lim - 0.8, lim + 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    frame_x0 = -lim - 0.7
    frame_y0 = -lim - 0.55
    draw_arrow_axes(ax,
                    x0=frame_x0, y0=frame_y0,
                    x1=lim + 0.4,  y1=lim + 0.75)

    # ── title ──────────────────────────────────────────────────────────────────
    fig.suptitle(
        "随机梯度下降法  Stochastic Gradient Descent",
        fontsize=14, color=TITLE_COLOR, x=0.43, y=0.99,
    )

    # ── colorbar ───────────────────────────────────────────────────────────────
    sm = ScalarMappable(norm=norm_flat, cmap=cmap_flat)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=ax, ticks=[], fraction=0.048, pad=0.02,
        boundaries=levels, drawedges=True,
    )
    cbar.outline.set_edgecolor("#888888")
    cbar.outline.set_linewidth(0.8)
    cbar.dividers.set_color("#888888")
    cbar.dividers.set_linewidth(0.8)
    cbar.ax.set_title("Cost", fontsize=13, fontweight="bold",
                      color="#111827", pad=10)

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
