"""Redraw of SVM_13: linearly inseparable two-class scatter plot.

Pure concept illustration for the Soft Margin SVM introduction.
- Red dots: ~7 points concentrated in upper-right region.
- Blue dots: ~7 points mainly in lower-left, but ONE outlier placed near
  the red cluster to make the data linearly inseparable.
- No separating line drawn (the whole point is that none exists).
- Axes shown only as arrows (no ticks, no labels, no title, no legend).
"""

import sys

sys.path.insert(0, "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/")

import _style

OUT_PATH = "/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_13.png"


def main():
    # Hard-coded coordinates (normalized roughly in [0.1, 0.9]).
    # Red cluster: upper-right
    red_x = [0.62, 0.70, 0.78, 0.68, 0.82, 0.74, 0.60]
    red_y = [0.65, 0.72, 0.60, 0.80, 0.73, 0.55, 0.78]

    # Blue cluster: lower-left main body + 1 intruder near red cluster
    blue_x = [0.22, 0.30, 0.18, 0.35, 0.25, 0.40,  0.72]
    blue_y = [0.25, 0.32, 0.40, 0.20, 0.45, 0.35,  0.38]
    # (0.72, 0.38) is the "outlier" blue point sitting inside the red region

    fig, ax = _style.new_ax(figsize=(5, 5))

    # Disable the default light grid from _style
    ax.grid(False)

    # Scatter: red class (upper-right)
    ax.scatter(red_x, red_y,
               color='#E53935', s=160, zorder=3,
               edgecolors='white', linewidths=0.8)

    # Scatter: blue class (lower-left + 1 intruder)
    ax.scatter(blue_x, blue_y,
               color='#1E88E5', s=160, zorder=3,
               edgecolors='white', linewidths=0.8)

    # Axis limits with room for arrows
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    # No ticks, no labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Hide all default spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw arrow axes using axes-fraction coordinates
    arrow_kw = dict(arrowstyle='->', color='#333333', lw=1.5,
                    mutation_scale=14)
    # x-axis: horizontal arrow at bottom
    ax.annotate('', xy=(0.97, 0.05), xytext=(0.03, 0.05),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=arrow_kw)
    # y-axis: vertical arrow at left
    ax.annotate('', xy=(0.05, 0.97), xytext=(0.05, 0.03),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=arrow_kw)

    _style.finalize(fig, OUT_PATH)
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
