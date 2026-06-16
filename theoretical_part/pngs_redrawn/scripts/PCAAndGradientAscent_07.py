"""Redraw of PCAAndGradientAscent_07: PCA projection concept sketch.

Two red arrows leave the origin in a wide V shape:
  - upper arrow: direction vector w = (w1, w2)
  - lower arrow: data point X^(i) = (X1^(i), X2^(i)), with a blue circle at tip

No axes / ticks / grid / frame -- only the sketch on a plain white canvas.
"""

import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')

import _style
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_PATH = '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/PCAAndGradientAscent_07.png'

# Palette from _style
ARROW_COLOR = _style.PALETTE[1]   # red #EF4444
POINT_COLOR = _style.PALETTE[0]   # blue #2563EB

# Hand-placed coordinates (pure concept, no data)
ORIGIN = (0.0, 0.0)
W_END  = (1.8, 1.2)    # direction vector w: upper-right
X_END  = (1.4, -0.6)   # data point X^(i): lower-right


def main():
    fig, ax = _style.new_ax(figsize=(6, 5))

    # --- Arrow for w (upper-right) ---
    ax.annotate(
        '',
        xy=W_END,
        xytext=ORIGIN,
        arrowprops=dict(
            arrowstyle='-|>',
            color=ARROW_COLOR,
            lw=2.5,
            mutation_scale=20,
        ),
        zorder=3,
    )

    # --- Arrow for X^(i) (lower-right) ---
    ax.annotate(
        '',
        xy=X_END,
        xytext=ORIGIN,
        arrowprops=dict(
            arrowstyle='-|>',
            color=ARROW_COLOR,
            lw=2.5,
            mutation_scale=20,
        ),
        zorder=3,
    )

    # --- Blue filled circle at X^(i) tip ---
    ax.scatter(
        [X_END[0]], [X_END[1]],
        s=150,
        color=POINT_COLOR,
        edgecolors='#1a4fa0',
        linewidths=1.0,
        zorder=5,
    )

    # --- Small arc hinting at the angle between the two vectors ---
    theta1 = np.degrees(np.arctan2(X_END[1], X_END[0]))   # lower angle
    theta2 = np.degrees(np.arctan2(W_END[1], W_END[0]))   # upper angle
    arc = mpatches.Arc(
        (0, 0), 0.45, 0.45,
        angle=0,
        theta1=theta1,
        theta2=theta2,
        color='#8B5CF6',
        lw=1.5,
        linestyle='--',
        zorder=2,
    )
    ax.add_patch(arc)

    # --- Labels ---
    # w label: right of arrow tip
    ax.text(
        W_END[0] + 0.07,
        W_END[1] + 0.04,
        r'$w = (w_1,\ w_2)$',
        fontsize=14,
        color='#222222',
        va='bottom',
        ha='left',
    )
    # X^(i) label: right-below arrow tip / blue dot
    ax.text(
        X_END[0] + 0.07,
        X_END[1] - 0.07,
        r'$X^{(i)} = (X_1^{(i)},\ X_2^{(i)})$',
        fontsize=14,
        color='#222222',
        va='top',
        ha='left',
    )

    # --- Clean canvas ---
    ax.set_xlim(-0.3, 2.8)
    ax.set_ylim(-1.1, 1.7)
    ax.axis('off')

    # Save directly (we already used _style for rcParams + fig creation)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved:', OUT_PATH)


if __name__ == '__main__':
    main()

