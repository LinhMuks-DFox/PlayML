import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')
import _style

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- Reproducible synthetic data ------------------------------------------
rng = np.random.default_rng(42)

# 10 base points in [0.1, 0.9] x-range
x_pts = np.sort(rng.uniform(0.08, 0.92, 10))
y_pts = 0.3 * np.sin(3 * np.pi * x_pts) + 0.15 * x_pts + rng.normal(0, 0.05, 10)
# Normalise y to [0.12, 0.88]
y_pts = (y_pts - y_pts.min()) / (y_pts.max() - y_pts.min()) * 0.76 + 0.12

# Smooth curve for the fitted output panels (in [0,1] panel coordinates)
x_curve = np.linspace(0.05, 0.95, 200)
y_curve = 0.3 * np.sin(3 * np.pi * x_curve) + 0.15 * x_curve
y_curve = (y_curve - y_curve.min()) / (y_curve.max() - y_curve.min()) * 0.76 + 0.12

# Per-panel sample weights (bottom panels A-D): from A→D some points grow
# We keep all panels using the same x_pts/y_pts but change point sizes.
# Base sizes
s_base = np.array([40, 40, 40, 40, 40, 40, 40, 40, 40, 40], dtype=float)

# Panel A: uniform weights
weights_A = s_base.copy()

# Panel B: points 3,7 "hard" → bigger
weights_B = s_base.copy()
weights_B[3] = 120
weights_B[7] = 110

# Panel C: points 3,7 even bigger, point 1 also grows
weights_C = s_base.copy()
weights_C[3] = 180
weights_C[7] = 160
weights_C[1] = 100

# Panel D: redistribute further
weights_D = s_base.copy()
weights_D[3] = 200
weights_D[7] = 190
weights_D[1] = 130
weights_D[5] = 90

# Opacity per panel (larger weight points stay opaque, lighter ones fade)
def make_alpha(w):
    a = w / w.max()
    return np.clip(a * 0.85 + 0.15, 0.2, 1.0)

alpha_A = make_alpha(weights_A)
alpha_B = make_alpha(weights_B)
alpha_C = make_alpha(weights_C)
alpha_D = make_alpha(weights_D)

# ---- Fit curves for output panels A' B' C' --------------------------------
# A': rough fit (offset)
y_fit_A = y_curve + 0.08 * np.sin(5 * np.pi * x_curve + 1.0)
# B': better
y_fit_B = y_curve + 0.04 * np.sin(4 * np.pi * x_curve + 0.5)
# C': nearly perfect
y_fit_C = y_curve + 0.015 * np.sin(3 * np.pi * x_curve)

# ---- Figure layout --------------------------------------------------------
fig = plt.figure(figsize=(13, 6.5), facecolor='white')

# Color palette from _style
BLUE   = _style.PALETTE[0]  # #2563EB
RED    = _style.PALETTE[1]  # #EF4444
AMBER  = _style.PALETTE[3]  # #F59E0B
VIOLET = _style.PALETTE[4]  # #8B5CF6

MODEL_FC  = '#D0E8FF'   # light blue fill for model boxes
PANEL_FC  = '#F5F5F5'   # light gray fill for data/output panels
PANEL_EC  = '#AAAAAA'   # panel edge color
ARROW_C   = '#666666'   # arrow color

# ---- Helper: draw a styled mini panel (returns Axes) ----------------------
def add_panel(fig, rect, label=None, label_pos='bottom_left'):
    """Add a rectangular panel with rounded border. Returns the Axes."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(PANEL_FC)
    for spine in ax.spines.values():
        spine.set_color(PANEL_EC)
        spine.set_linewidth(1.0)
    # Panel label
    if label is not None:
        if label_pos == 'bottom_left':
            ax.text(0.05, 0.07, label, transform=ax.transAxes,
                    fontsize=11, fontweight='bold', color='#333333',
                    ha='left', va='bottom', fontstyle='italic')
        else:  # top_left
            ax.text(0.05, 0.93, label, transform=ax.transAxes,
                    fontsize=11, fontweight='bold', color='#333333',
                    ha='left', va='top', fontstyle='italic')
    return ax

# ---- Helper: draw small coordinate arrows inside a panel ------------------
def draw_axes_arrows(ax, x0=0.07, y0=0.10, dx=0.30, dy=0.30):
    ax.annotate('', xy=(x0 + dx, y0), xytext=(x0, y0),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2))
    ax.annotate('', xy=(x0, y0 + dy), xytext=(x0, y0),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2))

# ---- Helper: draw model box (rectangle in figure coords) ------------------
def draw_model_box(fig, cx, cy, w=0.07, h=0.10, number='1'):
    """Draw a rounded-rectangle model box centered at (cx,cy) in figure coords."""
    ax_box = fig.add_axes([cx - w/2, cy - h/2, w, h])
    ax_box.set_xlim(0, 1)
    ax_box.set_ylim(0, 1)
    ax_box.set_xticks([])
    ax_box.set_yticks([])
    ax_box.set_facecolor(MODEL_FC)
    for spine in ax_box.spines.values():
        spine.set_color('#5B9BD5')
        spine.set_linewidth(1.5)
    # Gear + number
    ax_box.text(0.5, 0.62, '⚙', ha='center', va='center',
                fontsize=20, color='#2563EB', transform=ax_box.transAxes)
    ax_box.text(0.5, 0.22, number, ha='center', va='center',
                fontsize=12, fontweight='bold', color='#333333',
                transform=ax_box.transAxes)
    return ax_box

# ---- Helper: cross-axes arrow in figure coords ----------------------------
def fig_arrow(fig, x0, y0, x1, y1, color=ARROW_C, lw=1.5):
    arr = FancyArrowPatch(
        posA=(x0, y0), posB=(x1, y1),
        transform=fig.transFigure,
        arrowstyle='->', color=color,
        mutation_scale=12, linewidth=lw,
        connectionstyle='arc3,rad=0.0'
    )
    fig.add_artist(arr)

# ---- Layout constants ------------------------------------------------------
# Four columns: models 1-4 at these x-centres (in figure fraction)
# Panels: W=0.12, H=0.20 for data; H=0.22 for output
PW  = 0.13   # panel width
PHb = 0.22   # panel height (bottom data panels)
PHt = 0.22   # panel height (top output panels)
MBW = 0.08   # model box width
MBH = 0.12   # model box height

# x-centres for columns 1-4
col_x = [0.10, 0.34, 0.58, 0.82]
# y-centres for three rows (bottom, middle, top)
row_bot = 0.16   # data panels (centre y)
row_mid = 0.50   # model boxes
row_top = 0.80   # output panels

# ---- Bottom data panels (A, B, C, D) ---------------------------------------
panel_data = []
data_list  = [
    ('A',  x_pts, y_pts, weights_A, alpha_A),
    ('B',  x_pts, y_pts, weights_B, alpha_B),
    ('C',  x_pts, y_pts, weights_C, alpha_C),
    ('D',  x_pts, y_pts, weights_D, alpha_D),
]
for i, (label, xp, yp, ws, al) in enumerate(data_list):
    cx = col_x[i]
    rect = [cx - PW/2, row_bot - PHb/2, PW, PHb]
    ax = add_panel(fig, rect, label=label, label_pos='bottom_left')
    # Scatter with per-point size & alpha
    for j in range(len(xp)):
        ax.scatter(xp[j], yp[j], s=ws[j], color=BLUE, alpha=al[j],
                   edgecolors='white', linewidths=0.5, zorder=3)
    draw_axes_arrows(ax)
    panel_data.append(ax)

# ---- Model boxes (1, 2, 3, 4) at mid row ----------------------------------
for i, cx in enumerate(col_x):
    draw_model_box(fig, cx=cx, cy=row_mid, w=MBW, h=MBH, number=str(i+1))

# ---- Top output panels (A', B', C') — only 3 panels ----------------------
out_labels = ["A'", "B'", "C'"]
fit_curves  = [y_fit_A, y_fit_B, y_fit_C]
panel_out   = []
for i in range(3):
    cx = col_x[i]
    rect = [cx - PW/2, row_top - PHt/2, PW, PHt]
    ax = add_panel(fig, rect, label=out_labels[i], label_pos='top_left')
    # Scatter original points
    for j in range(len(x_pts)):
        ax.scatter(x_pts[j], y_pts[j], s=28, color=BLUE, alpha=0.65,
                   edgecolors='white', linewidths=0.4, zorder=3)
    # Fitted curve
    fc = fit_curves[i]
    ax.plot(x_curve, fc, color=RED, linewidth=2.0, zorder=4)
    draw_axes_arrows(ax)
    panel_out.append(ax)

# ---- Arrows: bottom panels → model boxes ----------------------------------
for i in range(4):
    cx = col_x[i]
    # bottom panel top edge → model box bottom edge
    bot_top  = row_bot + PHb/2
    mid_bot  = row_mid - MBH/2
    fig_arrow(fig, cx, bot_top, cx, mid_bot)

# ---- Arrows: model boxes → top output panels (only models 1-3) -----------
for i in range(3):
    cx = col_x[i]
    mid_top  = row_mid + MBH/2
    top_bot  = row_top - PHt/2
    fig_arrow(fig, cx, mid_top, cx, top_bot)

# ---- Title -----------------------------------------------------------------
fig.text(0.5, 0.97, 'AdaBoost — Sequential Ensemble',
         ha='center', va='top', fontsize=14, fontweight='bold', color='#222222')

# ---- Row annotations (right side) -----------------------------------------
fig.text(0.965, row_top,  'Output',  ha='right', va='center',
         fontsize=10, color='#555555', style='italic')
fig.text(0.965, row_mid,  'Model',   ha='right', va='center',
         fontsize=10, color='#555555', style='italic')
fig.text(0.965, row_bot,  'Data',    ha='right', va='center',
         fontsize=10, color='#555555', style='italic')

# ---- Weight-increase annotation -------------------------------------------
# Arrow beneath bottom panels showing data flows A→B→C→D
for i in range(3):
    x0 = col_x[i]   + PW/2 + 0.005
    x1 = col_x[i+1] - PW/2 - 0.005
    y_arr = row_bot
    fig_arrow(fig, x0, y_arr, x1, y_arr, color='#999999', lw=1.2)

# Small "weight ↑" label between A-B
fig.text((col_x[0] + col_x[1]) / 2, row_bot - 0.04,
         'reweight', ha='center', va='top', fontsize=8, color='#888888',
         style='italic')
fig.text((col_x[1] + col_x[2]) / 2, row_bot - 0.04,
         'reweight', ha='center', va='top', fontsize=8, color='#888888',
         style='italic')
fig.text((col_x[2] + col_x[3]) / 2, row_bot - 0.04,
         'reweight', ha='center', va='top', fontsize=8, color='#888888',
         style='italic')

# ---- Save ------------------------------------------------------------------
out_path = '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Ensemble_1.png'
# Use finalize but it calls tight_layout which can conflict with manual axes;
# so just save directly with the same dpi/bbox settings the style uses.
for ax in fig.get_axes():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {out_path}")
