import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')
import _style

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

# ---- reproducible random scatter points ------------------------------------
rng = np.random.default_rng(42)

# ============================================================
# Layout constants  (figure coords, y increases upward)
# ============================================================
FIG_W, FIG_H = 7, 6.5

# Row y-centers (in data / axes coords: we work in figure-fraction via transforms)
# We'll use a plain axes with xlim/ylim = [0,1] and axis off.

# y positions (fraction of axes height)
Y_TRAIN  = 0.10   # bottom: Training set panel
Y_SUBSET = 0.38   # middle: Subset 1 / Subset 2
Y_MODEL  = 0.72   # top: model boxes

# x positions
X_CENTER  = 0.50
X_SUB1    = 0.28
X_SUB2    = 0.72

# Model boxes x positions
X_M1 = 0.20
X_M2 = 0.50
X_M3 = 0.80

# ============================================================
# Helper: draw a parallelogram panel with scatter points
# ============================================================
def draw_parallelogram(ax, cx, cy, w, h, slant, facecolor, edgecolor,
                        n_pts=30, pt_color='#1E3A5F', label=None,
                        label_y_offset=-0.045, fontsize=10, label_color='#1E3A5F'):
    """Draw a parallelogram (slanted rectangle) with random scatter dots inside."""
    # 4 corners: bottom-left, bottom-right, top-right, top-left
    bl = [cx - w/2 + slant, cy - h/2]
    br = [cx + w/2 + slant, cy - h/2]
    tr = [cx + w/2 - slant, cy + h/2]
    tl = [cx - w/2 - slant, cy + h/2]
    poly = Polygon([bl, br, tr, tl], closed=True,
                   facecolor=facecolor, edgecolor=edgecolor,
                   linewidth=1.4, zorder=2)
    ax.add_patch(poly)

    # Random scatter inside the parallelogram (rejection sampling via bbox)
    xs_out, ys_out = [], []
    attempts = 0
    while len(xs_out) < n_pts and attempts < 5000:
        attempts += 1
        rx = rng.uniform(cx - w/2, cx + w/2)
        ry = rng.uniform(cy - h/2, cy + h/2)
        # linear interpolation of left/right x limits at height ry
        t = (ry - (cy - h/2)) / h  # 0 at bottom, 1 at top
        x_left  = (bl[0] * (1-t) + tl[0] * t)
        x_right = (br[0] * (1-t) + tr[0] * t)
        if x_left <= rx <= x_right:
            xs_out.append(rx)
            ys_out.append(ry)

    ax.scatter(xs_out, ys_out, s=8, color=pt_color, alpha=0.65,
               linewidths=0, zorder=3)

    if label:
        ax.text(cx, cy + h/2 + label_y_offset + 0.01,
                label, ha='center', va='bottom',
                fontsize=fontsize, color=label_color, fontweight='bold')


# ============================================================
# Helper: draw a fancy rounded-rectangle model box
# ============================================================
def draw_model_box(ax, cx, cy, w, h, facecolor, edgecolor,
                   title, subtitle, fontsize=10):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.015",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=1.6, zorder=4)
    ax.add_patch(box)
    # Title (model name)
    ax.text(cx, cy + 0.015, title,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='#1A1A2E', zorder=5)
    # Subtitle (small tree icon: simplified lines)
    ax.text(cx, cy - 0.035, subtitle,
            ha='center', va='center', fontsize=8,
            color='#4A4A6A', zorder=5)


# ============================================================
# Helper: curved/straight annotation arrow
# ============================================================
def arrow(ax, x0, y0, x1, y1, color='#333333',
          connectionstyle="arc3,rad=0.0", lw=1.6):
    ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->, head_width=0.25, head_length=0.012",
                    color=color,
                    lw=lw,
                    connectionstyle=connectionstyle,
                ),
                zorder=6)


# ============================================================
# Build figure
# ============================================================
fig, ax = _style.new_ax(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_aspect('auto')

# ---- 1. Training set panel (bottom) ----------------------------------------
draw_parallelogram(ax,
                   cx=X_CENTER, cy=Y_TRAIN,
                   w=0.50, h=0.12, slant=0.04,
                   facecolor='#D6EAF8', edgecolor='#2563EB',
                   n_pts=55, pt_color='#1A5276',
                   label=None)

# Label inside / below
ax.text(X_CENTER, Y_TRAIN + 0.005,
        '训练集  (Training set)',
        ha='center', va='center', fontsize=10.5,
        fontweight='bold', color='#1A5276', zorder=5)

# ---- 2. Subset panels (middle) ---------------------------------------------
# Subset 1
draw_parallelogram(ax,
                   cx=X_SUB1, cy=Y_SUBSET,
                   w=0.28, h=0.10, slant=0.03,
                   facecolor='#D5F5E3', edgecolor='#10B981',
                   n_pts=22, pt_color='#0E6655',
                   label=None)
ax.text(X_SUB1, Y_SUBSET + 0.002,
        '子集1  (Subset 1)',
        ha='center', va='center', fontsize=10,
        fontweight='bold', color='#0E6655', zorder=5)

# Subset 2
draw_parallelogram(ax,
                   cx=X_SUB2, cy=Y_SUBSET,
                   w=0.28, h=0.10, slant=0.03,
                   facecolor='#FDEBD0', edgecolor='#F59E0B',
                   n_pts=22, pt_color='#784212',
                   label=None)
ax.text(X_SUB2, Y_SUBSET + 0.002,
        '子集2  (Subset 2)',
        ha='center', va='center', fontsize=10,
        fontweight='bold', color='#784212', zorder=5)

# ---- 3. Model boxes (top) --------------------------------------------------
BOX_W, BOX_H = 0.20, 0.13

# Model 1 – blue
draw_model_box(ax, cx=X_M1, cy=Y_MODEL, w=BOX_W, h=BOX_H,
               facecolor='#AED6F1', edgecolor='#2471A3',
               title='模型 1\n(Model 1)',
               subtitle='决策树 / Decision Tree',
               fontsize=9.5)

# Model 2 – violet
draw_model_box(ax, cx=X_M2, cy=Y_MODEL, w=BOX_W, h=BOX_H,
               facecolor='#D7BDE2', edgecolor='#7D3C98',
               title='模型 2\n(Model 2)',
               subtitle='决策树 / Decision Tree',
               fontsize=9.5)

# Model 3 – pink/salmon (trained from Subset 2)
draw_model_box(ax, cx=X_M3, cy=Y_MODEL, w=BOX_W, h=BOX_H,
               facecolor='#F1948A', edgecolor='#CB4335',
               title='模型 3\n(Model 3)',
               subtitle='决策树 / Decision Tree',
               fontsize=9.5)

# ---- 4. Arrows: Training set → Subset 1 & Subset 2 ------------------------
# Training set top-center
TS_TOP_Y  = Y_TRAIN + 0.06   # top edge of training panel (approx)
SUB_BOT_Y = Y_SUBSET - 0.05  # bottom edge of subset panels

# left branch → Subset 1
arrow(ax, X_CENTER - 0.02, TS_TOP_Y,
         X_SUB1 + 0.01,  SUB_BOT_Y,
         color='#2563EB',
         connectionstyle="arc3,rad=-0.15")

# right branch → Subset 2
arrow(ax, X_CENTER + 0.02, TS_TOP_Y,
         X_SUB2 - 0.01,  SUB_BOT_Y,
         color='#2563EB',
         connectionstyle="arc3,rad=0.15")

# ---- 5. Arrows: Subsets → Models -------------------------------------------
SUB1_TOP_Y = Y_SUBSET + 0.05   # top of subset panels
SUB2_TOP_Y = Y_SUBSET + 0.05
MOD_BOT_Y  = Y_MODEL  - 0.065  # bottom of model boxes

# Subset 1 → Model 1
arrow(ax, X_SUB1, SUB1_TOP_Y,
         X_M1,   MOD_BOT_Y,
         color='#10B981',
         connectionstyle="arc3,rad=-0.08")

# Subset 1 → Model 2
arrow(ax, X_SUB1 + 0.01, SUB1_TOP_Y,
         X_M2 - 0.01,    MOD_BOT_Y,
         color='#10B981',
         connectionstyle="arc3,rad=0.12")

# Subset 2 → Model 3
arrow(ax, X_SUB2, SUB2_TOP_Y,
         X_M3,   MOD_BOT_Y,
         color='#F59E0B',
         connectionstyle="arc3,rad=0.08")

# ---- 6. Side labels: "分割 (Split)" and "训练 (Train)" --------------------
# "Split" label: near the midpoint of split arrows (between Training and Subsets)
split_y_mid = (TS_TOP_Y + SUB_BOT_Y) / 2
ax.text(0.03, split_y_mid,
        '分割\n(Split)',
        ha='left', va='center', fontsize=10,
        color='#2563EB', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#EBF5FB',
                  edgecolor='#2563EB', alpha=0.85))

# "Train" label: left of the arrows going up to models
train_y_mid = (SUB1_TOP_Y + MOD_BOT_Y) / 2
ax.text(0.03, train_y_mid,
        '训练\n(Train)',
        ha='left', va='center', fontsize=10,
        color='#10B981', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#EAFAF1',
                  edgecolor='#10B981', alpha=0.85))

# ---- 7. Title --------------------------------------------------------------
ax.set_title('Stacking 集成学习 — 训练流程\n(Stacking Ensemble: Training Procedure)',
             fontsize=13, fontweight='bold', color='#1A1A2E', pad=10)

# ---- Save ------------------------------------------------------------------
OUT = '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/Ensemble_4.png'
_style.finalize(fig, OUT)
