# Capítulo 13: gráficas de la teoría (VaR y ES sobre la densidad de pérdidas; Pareto generalizada)
import numpy as np
import estilo_graficos as eg
import matplotlib.pyplot as plt
from scipy import stats
from cap13_funciones import var_es_normal, var_es_t

p = 0.99; x = np.linspace(-4.5, 6.5, 800)
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4), sharey=True)
for a, (tit, dens, (v, e)) in zip(ax, [
        ("Normal estándar", stats.norm.pdf(x), var_es_normal(0, 1, p)),
        ("$t$ de Student con $\\nu=4$, varianza 1", stats.t.pdf(x * np.sqrt(2), 4) * np.sqrt(2), var_es_t(0, 1, 4, p))]):
    a.plot(x, dens, color=eg.AZUL, lw=1.4); k = x >= v
    a.fill_between(x[k], 0, dens[k], color=eg.LADRILLO, alpha=0.35)
    a.axvline(v, color=eg.LADRILLO, lw=1.2); a.axvline(e, color="black", lw=1.0, ls="--")
    a.set_title(f"{tit}\nVaR$_{{99}}$ = {v:.2f} (línea roja); ES$_{{99}}$ = {e:.2f} (discontinua)".replace(".", ","), fontsize=8.5)
    a.set_xlabel("pérdida $L$"); eg.coma(a)
    ins = a.inset_axes([0.62, 0.45, 0.36, 0.45]); kk = x >= 1.5
    ins.plot(x[kk], dens[kk], color=eg.AZUL, lw=1.2); ins.fill_between(x[k], 0, dens[k], color=eg.LADRILLO, alpha=0.45)
    ins.axvline(v, color=eg.LADRILLO, lw=1); ins.axvline(e, color="black", lw=0.9, ls="--")
    ins.set_ylim(0, 0.06); ins.tick_params(labelsize=6); ins.set_title("cola ampliada: área 1 %", fontsize=7); eg.coma(ins)
    print(f"{tit}: VaR99 = {v:.4f}, ES99 = {e:.4f}")
ax[0].set_ylabel("densidad")
fig.tight_layout(); fig.savefig("graficos/cap13_fig_var_es.pdf")

y = np.linspace(0, 8, 400); fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
for xi, col, ls in [(-0.2, eg.OLIVA, ":"), (0.0, eg.AZUL, "-"), (0.3, eg.LADRILLO, "--")]:
    d = stats.genpareto.pdf(y, xi, scale=1.0)
    ax[0].semilogy(y, np.where(d > 0, d, np.nan), color=col, ls=ls, lw=1.5, label=f"$\\xi={xi}$".replace(".", ","))
    v = np.linspace(0, 4.5, 50); ax[1].plot(v, (1 + xi * v) / (1 - xi), color=col, ls=ls, lw=1.5, label=f"$\\xi={xi}$".replace(".", ","))
ax[0].set_ylim(1e-4, 1.2); ax[0].set_title("Densidad de la Pareto generalizada ($\\beta=1$)", fontsize=9); ax[0].set_xlabel("exceso $y$")
ax[0].yaxis.set_major_formatter(plt.LogFormatterMathtext()); ax[0].legend(fontsize=8)
ax[1].set_title("Media de los excesos $e(v)=\\mathrm{E}[L-v\\mid L>v]$", fontsize=9); ax[1].set_xlabel("umbral $v$ (desde $u$)")
ax[1].legend(fontsize=8)
for a in ax: eg.coma(a)
ax[0].yaxis.set_major_formatter(plt.LogFormatterMathtext())
fig.tight_layout(); fig.savefig("graficos/cap13_fig_gpd.pdf")
