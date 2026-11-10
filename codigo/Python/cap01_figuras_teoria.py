# Capítulo 1: gráficas de la teoría (no usan datos)
import numpy as np
from scipy import stats
import estilo_graficos as eg
import matplotlib.pyplot as plt

# --- Figura: densidad lognormal de 1+R, con media y mediana ---
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
mu = 0.08
x = np.linspace(0.01, 3.2, 600)
for sigma, col, ls in [(0.15, eg.GRIS, ":"), (0.30, eg.AZUL, "-"), (0.50, eg.LADRILLO, "--")]:
    ax[0].plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu)), color=col, ls=ls,
               label=rf"$\sigma={sigma:.2f}$".replace(".", ","))
ax[0].set_xlabel("rendimiento bruto $1+R$"); ax[0].set_ylabel("densidad")
ax[0].set_title(r"Lognormal con $\mu=0{,}08$ y distintas $\sigma$"); ax[0].legend()
s = 0.5; dens = stats.lognorm(s=s, scale=np.exp(mu))
ax[1].plot(x, dens.pdf(x), color=eg.LADRILLO)
ax[1].fill_between(x, dens.pdf(x), color=eg.LADRILLO, alpha=0.12)
for v, nom, col, ls in [(np.exp(mu - s**2), "moda", eg.GRIS, ":"), (np.exp(mu), "mediana $e^{\\mu}$", eg.AZUL, "--"),
                        (np.exp(mu + s**2 / 2), "media $e^{\\mu+\\sigma^2/2}$", eg.OLIVA, "-")]:
    ax[1].axvline(v, color=col, ls=ls, lw=1.2, label=f"{nom} = {v:.3f}".replace(".", ","))
ax[1].set_xlabel("rendimiento bruto $1+R$"); ax[1].set_title(r"$\sigma=0{,}50$: moda < mediana < media")
ax[1].legend(fontsize=8)
for a in ax: eg.coma(a)
fig.savefig("graficos/cap01_fig_lognormal.pdf")

# --- Figura: razón de varianzas Var[r(k)]/(k sigma^2) con autocorrelación AR(1) ---
fig, ax = plt.subplots(figsize=(6, 3.3))
k = np.arange(1, 61)
for phi, col, ls in [(0.2, eg.LADRILLO, "-"), (0.1, eg.OLIVA, "--"), (0.0, "black", "-"),
                     (-0.1, eg.GRIS, "--"), (-0.2, eg.AZUL, "-")]:
    vr = (k * (1 - phi**2) - 2 * phi * (1 - phi**k)) / (1 - phi)**2 / k
    ax.plot(k, vr, color=col, ls=ls, label=rf"$\rho_1={phi:+.1f}$".replace(".", ","))
ax.set_xlabel("horizonte $k$ (períodos)"); ax.set_ylabel(r"$\mathrm{Var}[r_t(k)]\,/\,(k\sigma^2)$")
ax.legend(ncol=1, fontsize=8, loc="center right"); eg.coma(ax)
fig.savefig("graficos/cap01_fig_razonvar.pdf")
