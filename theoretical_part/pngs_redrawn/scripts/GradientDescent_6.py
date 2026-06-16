"""Redraw of GradientDescent_6: non-convex loss curve with local and global minima.

Pure concept diagram -- no ticks, no numeric labels, no grid.
Left (deeper) valley = global minimum.  Middle (shallower) valley = local minimum.
Right side rises monotonically.  A single italic 'f' label sits near the global
minimum valley bottom, matching the hand-drawn original.

Curve is constructed from a hand-tuned polynomial so that:
  * global minimum (left valley) is clearly lower than the local minimum (right valley)
  * the right side continues to rise after the local minimum
  * the curve is smooth and visually clean

Style: unified _style.py (deep indigo curve, CJK fonts, white background).
"""

import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')

import numpy as np
import _style

OUT_PATH = '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/GradientDescent_6.png'

# Deep indigo/purple as specified -- close to #4B0082 (indigo)
CURVE_COLOR = '#4B0082'


def f(x):
    """Non-convex double-valley loss function.

    Hand-tuned so the left valley is the global (deeper) minimum and
    the middle valley is a shallower local minimum, with the right end
    rising monotonically.
    """
    return (0.08 * x**4
            - 0.35 * x**3
            - 0.55 * x**2
            + 2.2 * x
            + 3.5)


def main():
    x = np.linspace(-2.8, 5.2, 600)
    y = f(x)

    fig, ax = _style.new_ax(figsize=(6, 3.5))

    # Main curve: deep indigo, linewidth ~2.5
    ax.plot(x, y, color=CURVE_COLOR, linewidth=2.5,
            solid_capstyle='round', zorder=2)

    # Locate the global minimum (left basin) numerically for label placement
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(f, bounds=(-2.8, 0.5), method='bounded')
    gx, gy = res.x, res.fun

    # Italic 'f' label near the global minimum (slightly left and above bottom)
    ax.text(gx - 0.55, gy + 0.5, r'$f$',
            fontstyle='italic', fontsize=14, color='#222222',
            ha='center', va='center')

    # --- Minimal schematic style ---
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # Hide all four spines for a completely frameless look (pure concept diagram)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Provide a little vertical padding so the curve doesn't touch the edges
    y_margin = (y.max() - y.min()) * 0.12
    ax.set_ylim(y.min() - y_margin, y.max() + y_margin)
    ax.set_xlim(x.min(), x.max())

    _style.finalize(fig, OUT_PATH)
    print('saved:', OUT_PATH)


if __name__ == '__main__':
    main()
