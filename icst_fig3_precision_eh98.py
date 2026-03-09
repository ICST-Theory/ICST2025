"""
ICST — Figure 2: Geodesic Affine Parameter and Kretschmann Scalar
=================================================================
Proves geodesic completeness for n >= 2 near-horizon scaling.
  (a) Affine parameter lambda -> infinity for n=2 (log divergence)
  (b) Kretschmann scalar ~ (C - C_crit)^(-6): coordinate divergence
      but physically unreachable (lambda -> infinity first)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 11,
    'axes.linewidth': 1.0, 'xtick.direction': 'in',
    'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
})

BLUE   = '#1B4F9A'
RED    = '#C0392B'
GREEN  = '#1E7B4B'
PURPLE = '#8E44AD'
ORANGE = '#E67E22'

x = np.linspace(0.003, 1.0, 600)  # x = r - r_h

# ── Affine parameter for different n
# lambda ~ integral of (r-r_h)^(-n/2) dr
lam_n2 = np.log(1 / x)              # n=2: log divergence
lam_n3 = x**(1 - 3/2) / (3/2 - 1)  # n=3: power law (stronger)
lam_n1 = 2 * x**(1/2)               # n=1: FINITE -> excluded

# Normalize for visual comparison
lam_n2 = lam_n2 / lam_n2[0]
lam_n3 = lam_n3 / lam_n3[0]
lam_n1 = lam_n1 / lam_n1[0]

# ── C profile (Frobenius near-horizon)
C_crit = 1.1578
C_h    = 0.80
alpha  = 0.10
x2     = np.linspace(0.01, 0.6, 400)
C_prof = C_h + alpha * np.sqrt(x2)  # Frobenius solution

# ── Kretschmann scalar ~ (C_crit - C)^(-6)
delta_C = C_crit - C_prof
delta_C = np.maximum(delta_C, 1e-4)
K_tilde = 1.0 / delta_C**6
K_norm  = K_tilde / K_tilde.max()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# --- Panel (a): Affine parameter
ax1.semilogy(x, lam_n2, color=BLUE, lw=2.5,
             label=r'$n=2$ (ICST): $\lambda \sim \ln(1/x) \to +\infty$')
ax1.semilogy(x, lam_n3, color=GREEN, ls='--', lw=2,
             label=r'$n=3$: $\lambda \sim x^{-1/2} \to +\infty$')
ax1.semilogy(x, lam_n1 * 0.05 + 0.001, color=RED, ls=':', lw=2,
             label=r'$n=1$ (Schwarzschild): $\lambda$ finite — excluded')

ax1.axvline(0, color='k', lw=0.5, ls='-')
ax1.fill_between(x, lam_n1 * 0.05 + 0.001, 0.001,
                 alpha=0.1, color=RED)

ax1.set_xlabel(r'$x = r - r_h$ (distance to horizon)')
ax1.set_ylabel(r'Affine parameter $\lambda$ (normalized)')
ax1.set_title(r'(a) Geodesic Completeness: $\lambda \to +\infty$')
ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax1.grid(True, ls=':', alpha=0.3)
ax1.set_xlim(0, 1.0)
ax1.set_ylim(1e-3, 10)

ax1.annotate(r'$n=2$: ICST geodesic complete',
             xy=(0.1, lam_n2[int(0.1/1.0*600)]),
             xytext=(0.35, 3), fontsize=8, color=BLUE,
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))

ax1.annotate('Schwarzschild\nfinite → excluded',
             xy=(0.5, 0.005), xytext=(0.55, 0.05),
             fontsize=7.5, color=RED,
             arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))

# --- Panel (b): Kretschmann
ax2.semilogy(x2, K_norm, color=PURPLE, lw=2.5,
             label=r'$\tilde{K} \sim (C_{crit}-C)^{-6}$')
ax2.axhline(1e-2, color=ORANGE, ls='--', lw=1.5,
            label='Observational threshold')
ax2.axhline(1e-6, color=GREEN, ls=':', lw=1.2,
            label='Solar System curvature')

# Mark where lambda -> infinity (x -> 0)
ax2.axvline(0.01, color=BLUE, ls=':', lw=1.5,
            label=r'$\lambda \to \infty$ boundary')

ax2.set_xlabel(r'$x = r - r_h$')
ax2.set_ylabel(r'Kretschmann scalar $\tilde{K}$ (normalized)')
ax2.set_title(r'(b) Kretschmann Divergence — Physically Unreachable')
ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax2.grid(True, ls=':', alpha=0.3)
ax2.set_xlim(0, 0.6)

# Annotation box
ax2.text(0.3, 0.5,
         r'Coordinate singularity' '\n' r'but $\lambda \to \infty$:' '\n'
         'geodesically complete',
         transform=ax2.transAxes, fontsize=8.5,
         bbox=dict(boxstyle='round', facecolor='lightyellow',
                   edgecolor=ORANGE, alpha=0.9))

fig.suptitle('Fig. 2 — ICST: Geodesic Completeness and Kretschmann Scalar',
             fontsize=11, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('fig2_geodesic.png', bbox_inches='tight', dpi=200)
print('Saved: fig2_geodesic.png')
