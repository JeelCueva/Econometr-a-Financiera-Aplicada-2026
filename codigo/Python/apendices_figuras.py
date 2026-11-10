"""Figuras teóricas del Apéndice A (sin datos): densidades normal, t, chi-cuadrado y F, y el TCL."""
import numpy as np
from scipy import stats
import estilo_graficos as eg
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0))
x = np.linspace(-5, 5, 600)
ax[0].plot(x, stats.norm.pdf(x), color="black", lw=1.3, label="$N(0,1)$")
for g, c in [(1, eg.LADRILLO), (3, eg.AZUL), (10, eg.OLIVA)]:
    ax[0].plot(x, stats.t.pdf(x, g), color=c, lw=1, ls="--" if g == 1 else "-", label=f"$t_{{{g}}}$")
ax[0].set_title("Normal y $t$ de Student", loc="left"); ax[0].legend(fontsize=7.5)
x = np.linspace(0.01, 20, 600)
for k, c in [(1, eg.LADRILLO), (3, eg.AZUL), (5, eg.OLIVA), (10, eg.GRIS)]:
    ax[1].plot(x, stats.chi2.pdf(x, k), color=c, lw=1, label=f"$\\chi^2_{{{k}}}$")
ax[1].set_ylim(0, 0.5); ax[1].set_title("Chi-cuadrado", loc="left"); ax[1].legend(fontsize=7.5)
x = np.linspace(0.01, 5, 600)
for (m, n), c in [((2, 10), eg.LADRILLO), ((5, 10), eg.AZUL), ((10, 30), eg.OLIVA), ((30, 120), eg.GRIS)]:
    ax[2].plot(x, stats.f.pdf(x, m, n), color=c, lw=1, label=f"$F_{{{m},{n}}}$")
ax[2].set_title("$F$ de Snedecor", loc="left"); ax[2].legend(fontsize=7.5)
for a in ax: eg.coma(a)
fig.savefig("graficos/apA_fig_distribuciones.pdf")

# Teorema central del límite: media de T rendimientos exponenciales centrados (asimétricos)
rng = np.random.default_rng(2026)
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.8), sharey=True)
z = np.linspace(-4, 4, 400)
for a, T in zip(ax, [1, 5, 50]):
    m = rng.exponential(1.0, size=(50000, T)).mean(axis=1)
    zt = (m - 1) * np.sqrt(T)                       # sqrt(T)(media - mu)/sigma con mu = sigma = 1
    a.hist(zt, bins=np.arange(-4, 6, 0.2), density=True, color=eg.AZUL, alpha=0.7, label="simulación")
    a.plot(z, stats.norm.pdf(z), color=eg.LADRILLO, lw=1.2, label="$N(0,1)$")
    a.set_title(f"$T={T}$", loc="left"); a.set_xlim(-4, 5); eg.coma(a)
ax[0].legend(fontsize=7.5)
fig.savefig("graficos/apA_fig_tcl.pdf")
