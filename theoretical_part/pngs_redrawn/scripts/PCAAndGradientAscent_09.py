"""重绘 PCAAndGradientAscent_09.

PCA 向量分解图：X'^(i) = X^(i) - X_{project}^(i)

几何关系（平行四边形法则）：
  X^(i) = X_{project}^(i) + X'^(i)
  即：X_pt = X_proj + X_res，其中 X_res = X_pt - X_proj

几何参数（第 3 轮修正）：
  w 方向角 ~50°（较大仰角，使 X_proj 斜度更陡，视觉接近原图）
  perp_hat = [sin θ, -cos θ]，即顺时针旋转 90°，指向右下方
  X_pt = X_proj + perp_len * perp_hat
      → X_res = X_pt - X_proj = perp_len * perp_hat，方向为【右下】（正x，负y）✓
  画面布局：O 在画布中左，三向量+平行四边形占 60-70% 画布面积

平行四边形四个顶点（全部 numpy 精确计算）：
  O       = (0, 0)
  X_proj  = O + proj_len * w_hat              （蓝色箭头终点，w 轴上）
  X_res   = perp_len * perp_hat               （绿色箭头终点，垂直于 w）
  X_pt    = X_proj + X_res = X_proj + X_res   （灰色箭头终点，即 X^(i)）

  验证 X_pt = X_proj + X_res 严格成立（assert）
  验证 dot(X_res, w_hat) ≈ 0（assert）
  验证 X_proj = (X_pt · w_hat) * w_hat（assert，X_proj 是 X_pt 在 w 上的正射影）

辅助元素：
  红色射线：O → w_tip（w 轴方向，超出 X_proj 延伸）
  蓝色箭头：O → X_proj（投影向量）
  灰色箭头：O → X_pt（原始数据向量 X^(i)）
  绿色箭头：O → X_res（残差 X'^(i)，方向为右下，正x负y）
  虚线平行四边形边：X_res↔X_pt（平行于 O→X_proj）, X_proj↔X_pt（平行于 O→X_res）
  点线：X_pt → X_proj（从数据点垂直落到 w 轴，展示投影）
  直角标记：X_proj 处，边长充分大，清晰可见
"""

import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')
import _style

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# 几何参数（全部 numpy 精确计算）
# ─────────────────────────────────────────────────────────────────────────────

# w 方向角：50°（较大仰角，使图形更接近原图视觉——X_proj 偏陡，X_res 明显右下）
theta = np.deg2rad(50.0)

# w 单位向量（已归一化）
w_hat = np.array([np.cos(theta), np.sin(theta)])

# perp_hat = 顺时针旋转 90°，方向 = [sin θ, -cos θ]
# 50° 时：[sin50°, -cos50°] = [+0.766, -0.643]，即【正x，负y】= 右下 ✓
perp_hat = np.array([np.sin(theta), -np.cos(theta)])

# 核验 perp_hat ⊥ w_hat
assert abs(np.dot(perp_hat, w_hat)) < 1e-12, "perp_hat 未垂直于 w_hat"
assert abs(np.linalg.norm(perp_hat) - 1.0) < 1e-12, "perp_hat 未归一化"

# 原点
O = np.array([0.0, 0.0])

# ─────────────────────────────────────────────────────────────────────────────
# 关键坐标（全部由公式显式推导）
# ─────────────────────────────────────────────────────────────────────────────

proj_len = 3.2   # 投影向量长度（沿 w 方向）
perp_len = 2.4   # 残差向量长度（垂直于 w，即 |X'^(i)|）

# [公式] X_{project}^(i) 终点
X_proj = O + proj_len * w_hat

# [公式] X'^(i) 终点（自由向量，从 O 出发）= perp_len * perp_hat
# 方向为 [sin50°, -cos50°] = [+0.766, -0.643]，即右下方 ✓
X_res = perp_len * perp_hat

# [公式] X^(i) 终点 = X_proj + X_res（平行四边形法则）
X_pt = X_proj + X_res

# ─────────────────────────────────────────────────────────────────────────────
# 精确验证（任何一个 assert 失败则报错，保证几何正确）
# ─────────────────────────────────────────────────────────────────────────────

# 验证1：向量加法闭合（平行四边形法则）
assert np.allclose(X_pt, X_proj + X_res, atol=1e-12), \
    f"平行四边形闭合失败: X_proj+X_res={X_proj+X_res}, X_pt={X_pt}"

# 验证2：残差垂直于 w
dot_val = np.dot(X_res, w_hat)
assert abs(dot_val) < 1e-10, f"X_res 不垂直于 w: dot={dot_val}"

# 验证3：X_proj 确实是 X_pt 在 w 上的正射影
proj_scalar = np.dot(X_pt, w_hat)
X_proj_verify = proj_scalar * w_hat
assert np.allclose(X_proj, X_proj_verify, atol=1e-10), \
    f"投影验证失败: X_proj={X_proj}, 重算={X_proj_verify}"

# 验证4：X_res 方向（正x，负y）
assert X_res[0] > 0 and X_res[1] < 0, \
    f"X_res 应为右下方向（正x，负y），实际={X_res}"

# ─────────────────────────────────────────────────────────────────────────────
# 打印坐标摘要（帮助核对）
# ─────────────────────────────────────────────────────────────────────────────
print("=== 几何验证 ===")
print(f"  theta    = {np.rad2deg(theta):.1f}°")
print(f"  w_hat    = {w_hat.round(4)}")
print(f"  perp_hat = {perp_hat.round(4)}  (应为正x,负y = 右下)")
print(f"  O        = {O}")
print(f"  X_proj   = {X_proj.round(4)}")
print(f"  X_res    = {X_res.round(4)}  (应为右下方向)")
print(f"  X_pt     = {X_pt.round(4)}  = X_proj + X_res")
print(f"  dot(X_res, w_hat) = {dot_val:.2e}  (应≈0)")
print(f"  proj_scalar = {proj_scalar:.4f}  (应== proj_len={proj_len})")

# ─────────────────────────────────────────────────────────────────────────────
# w 轴射线端点（超出 X_proj 延伸，强化"沿 w 轴投影"含义）
# ─────────────────────────────────────────────────────────────────────────────
w_extend = 1.2
w_tip = O + (proj_len + w_extend) * w_hat

# ─────────────────────────────────────────────────────────────────────────────
# 图形初始化
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = _style.new_ax(figsize=(7.5, 6.0))
ax.set_aspect('equal')
ax.axis('off')

# 颜色常量
RED        = '#EF4444'
BLUE       = '#2563EB'
GREEN      = '#10B981'
GRAY       = '#6B7280'
BLACK      = '#1a1a1a'
DASH_COLOR = '#444444'

# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def draw_arrow(start, end, color, lw=2.4, ms=18, zo=4, ls='solid'):
    """从 start 到 end 绘制带箭头的线段。"""
    ax.annotate(
        "",
        xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            linewidth=lw,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=ms,
            linestyle=ls,
        ),
        zorder=zo,
    )


def right_angle_mark(corner, along, perp, size=0.22, color=BLACK, zo=7):
    """在 corner 处绘制直角标记（小方块），边长为 size（数据坐标单位）。

    along: 沿 w 方向的单位向量（函数内部归一化）
    perp:  垂直方向单位向量（朝向 X^(i) 那侧，函数内部归一化）
    size:  方块边长（单位与数据坐标一致，较大值保证可见）
    """
    d1 = np.array(along, dtype=float)
    d1 /= np.linalg.norm(d1)
    d2 = np.array(perp, dtype=float)
    d2 /= np.linalg.norm(d2)
    # 四个顶点（从 corner 出发，沿 d1 和 d2 各走 size）
    p1 = corner + d1 * size
    p2 = corner + d1 * size + d2 * size
    p3 = corner + d2 * size
    pts = np.array([corner, p1, p2, p3, corner])
    ax.plot(
        pts[:, 0], pts[:, 1],
        color=color, linewidth=2.0, zorder=zo,
        solid_capstyle='round', solid_joinstyle='round',
    )

# ─────────────────────────────────────────────────────────────────────────────
# 绘制元素（从低 zorder 到高 zorder）
# ─────────────────────────────────────────────────────────────────────────────

# 1. w 轴射线（细线 + 箭头，从 O 出发经 X_proj 延伸至 w_tip）
ax.plot(
    [O[0], w_tip[0]], [O[1], w_tip[1]],
    color=RED, linewidth=1.4, linestyle='-',
    zorder=2, alpha=0.50,
)
# 箭头仅在 w_tip 端绘制（在延伸段上）
draw_arrow(
    O + (proj_len + 0.4) * w_hat, w_tip,
    RED, lw=2.0, ms=16, zo=3,
)

# 2. 平行四边形虚线边（补全平行四边形，展示向量加法几何意义）
#    边 a：X_res 终点 → X_pt 终点（平行于 O→X_proj）
ax.plot(
    [X_res[0], X_pt[0]], [X_res[1], X_pt[1]],
    color=DASH_COLOR, linewidth=1.5, linestyle='--',
    zorder=2, alpha=0.70,
)
#    边 b：X_proj 终点 → X_pt 终点（平行于 O→X_res）
ax.plot(
    [X_proj[0], X_pt[0]], [X_proj[1], X_pt[1]],
    color=DASH_COLOR, linewidth=1.5, linestyle='--',
    zorder=2, alpha=0.70,
)

# 3. 从 X^(i) 到投影脚的垂线（点线，展示投影操作）
ax.plot(
    [X_pt[0], X_proj[0]], [X_pt[1], X_proj[1]],
    color=DASH_COLOR, linewidth=1.8, linestyle=':',
    zorder=3, alpha=0.85,
)

# 4. 直角标记（在投影脚 X_proj 处）
#    along = -w_hat（朝向 O 方向，方块在 w 轴 O 侧）
#    perp  = perp_hat（朝向 X_pt，即 X^(i) 所在方向）
#    size  = 0.22 数据坐标单位，足够清晰
right_angle_mark(
    X_proj,
    along=-w_hat,
    perp=perp_hat,
    size=0.22,
    color=BLACK,
    zo=7,
)

# 5. 三条主向量箭头（全部从 O 出发）
#    灰色：O → X^(i)
draw_arrow(O, X_pt, GRAY, lw=2.2, ms=17, zo=4)
#    蓝色：O → X_{project}^(i)
draw_arrow(O, X_proj, BLUE, lw=2.8, ms=18, zo=5)
#    绿色：O → X'^(i)（残差，右下方向）
draw_arrow(O, X_res, GREEN, lw=2.8, ms=18, zo=5)

# 6. 关键点标记
ax.plot(*O,      'o', color=BLACK, markersize=7,  zorder=8)
ax.plot(*X_pt,   'o', color=GRAY,  markersize=8,  zorder=7,
        markeredgecolor='white', markeredgewidth=1.2)
ax.plot(*X_proj, 's', color=BLUE,  markersize=6,  zorder=7,
        markeredgecolor='white', markeredgewidth=1.0)
ax.plot(*X_res,  'D', color=GREEN, markersize=6,  zorder=7,
        markeredgecolor='white', markeredgewidth=1.0)

# ─────────────────────────────────────────────────────────────────────────────
# 标签
# ─────────────────────────────────────────────────────────────────────────────
fs_main = 12
fs_sub  = 11

# O 标签
ax.text(
    O[0] - 0.10, O[1] - 0.07,
    r'$O$',
    color=BLACK, fontsize=fs_main, ha='right', va='top', zorder=9,
)

# w 标签（射线末端右侧）
ax.annotate(
    r'$w=(w_1,\,w_2)$',
    xy=w_tip,
    xytext=(8, 4), textcoords='offset points',
    color=RED, fontsize=fs_main, ha='left', va='bottom', zorder=9,
)

# X^(i) 标签（右侧，数据点右边）
ax.annotate(
    r'$X^{(i)}=(X^{(i)}_1,\,X^{(i)}_2)$',
    xy=X_pt,
    xytext=(10, 2), textcoords='offset points',
    color=GRAY, fontsize=fs_sub, ha='left', va='center', zorder=9,
)

# X_{project}^(i) 标签（投影点左上方）
ax.annotate(
    r'$X_{project}^{(i)}=(X^{(i)}_{pr1},\,X^{(i)}_{pr2})$',
    xy=X_proj,
    xytext=(-8, 10), textcoords='offset points',
    color=BLUE, fontsize=fs_sub, ha='right', va='bottom', zorder=9,
)

# X'^(i) 标签（残差终点右下方，避免与 w 轴重叠）
ax.annotate(
    r"$X^{\prime(i)}=(X^{\prime(i)}_1,\,X^{\prime(i)}_2)$",
    xy=X_res,
    xytext=(8, -12), textcoords='offset points',
    color=GREEN, fontsize=fs_sub, ha='left', va='top', zorder=9,
)

# ─────────────────────────────────────────────────────────────────────────────
# 轴范围（使图形占 60-70% 画布；O 置于中左位置）
# ─────────────────────────────────────────────────────────────────────────────
# 收集所有关键点
all_pts = np.vstack([O, w_tip, X_pt, X_proj, X_res])
xmin_d = all_pts[:, 0].min()   # 数据坐标下的 xmin（可能为 0，即 O 和 X_res 是正x）
xmax_d = all_pts[:, 0].max()
ymin_d = all_pts[:, 1].min()   # 数据坐标下的 ymin（X_res 有负y）
ymax_d = all_pts[:, 1].max()

# 四周留白（为标签预留，不过度宽松）
pad_left   = 0.60   # O 标签 + 左侧余量
pad_right  = 2.10   # X^(i) 坐标文本（较长）
pad_top    = 0.55   # w 标签
pad_bottom = 0.90   # X_res 标签（向右下偏移，但还需留白）

ax.set_xlim(xmin_d - pad_left,  xmax_d + pad_right)
ax.set_ylim(ymin_d - pad_bottom, ymax_d + pad_top)

# ─────────────────────────────────────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────────────────────────────────────
_style.finalize(
    fig,
    '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/PCAAndGradientAscent_09.png',
)
print("Saved -> PCAAndGradientAscent_09.png")
