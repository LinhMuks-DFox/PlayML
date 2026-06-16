import sys
sys.path.insert(0, '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/')
import _style

import numpy as np

OUT_PATH = '/Users/mux/code_workspace/PlayML/theoretical_part/pngs_redrawn/SVM_18.png'

# Normal (Gaussian) PDF
def gauss_pdf(x, mu, sigma2):
    return (1.0 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))

x = np.linspace(-5, 5, 500)

# Four parameter sets: (mu, sigma2, color, label)
curves = [
    (0,  0.2, _style.PALETTE[0], r'$\mu=0,\;\sigma^2=0.2$'),
    (0,  1.0, _style.PALETTE[1], r'$\mu=0,\;\sigma^2=1.0$'),
    (0,  5.0, _style.PALETTE[3], r'$\mu=0,\;\sigma^2=5.0$'),
    (-2, 0.5, _style.PALETTE[2], r'$\mu=-2,\;\sigma^2=0.5$'),
]

fig, ax = _style.new_ax(figsize=(7, 4.5))

for mu, sigma2, color, label in curves:
    y = gauss_pdf(x, mu, sigma2)
    ax.plot(x, y, color=color, label=label)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\varphi_{\mu,\sigma^2}(x)$')
ax.set_title('正态分布 PDF：均值与方差对形状的影响')
ax.set_xlim(-5, 5)
ax.set_ylim(bottom=0)
ax.legend(loc='upper right')

# Optional annotation about gamma and sigma relationship
ax.text(0.98, 0.60,
        r'$\gamma \propto 1/\sigma^2$' + '\n'
        r'$\gamma$ 越大 → 曲线越窄' + '\n'
        r'$\gamma$ 越小 → 曲线越宽',
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=9,
        color='#64748B',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#CCCCCC', alpha=0.85))

_style.finalize(fig, OUT_PATH)
print(f'Saved to {OUT_PATH}')
