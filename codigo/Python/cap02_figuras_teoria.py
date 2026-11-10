# Capítulo 2: gráficas de la teoría (no usan datos; las simulaciones tienen semilla fija)
import numpy as np
from scipy import stats
import estilo_graficos as eg
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 801)

# --- Figura: asimetría y curtosis ---
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
for a_, col, ls, nom in [(-5, eg.LADRILLO, "--", "asimetría negativa"), (0, "black", "-", "simétrica (normal)"),
                         (5, eg.AZUL, ":", "asimetría positiva")]:
    d = stats.skewnorm(a_); z = (x * d.std() + d.mean())          # misma media y varianza
    ax[0].plot(x, d.pdf(z) * d.std(), color=col, ls=ls, label=f"{nom}: $S={float(d.stats(moments='s')):+.2f}$".replace(".", ","))
ax[0].set_title("Asimetría (media 0 y varianza 1)"); ax[0].set_xlabel("$z$"); ax[0].set_ylabel("densidad")
ax[0].legend(fontsize=7.5, loc="upper left"); ax[0].set_ylim(0, 0.62)
nu = 5; esc = np.sqrt(nu / (nu - 2))
t_std = stats.t.pdf(x * esc, nu) * esc                              # t_5 reescalada a varianza 1
ax[1].plot(x, stats.norm.pdf(x), color="black", label="normal: $K=3$")
ax[1].plot(x, t_std, color=eg.LADRILLO, ls="--", label="$t_5$ con varianza 1: $K=9$")
ax[1].fill_between(x, stats.norm.pdf(x), t_std, where=t_std < stats.norm.pdf(x), color=eg.AZUL, alpha=0.18)
ax[1].fill_between(x, stats.norm.pdf(x), t_std, where=t_std > stats.norm.pdf(x), color=eg.LADRILLO, alpha=0.18)
ax[1].annotate("pico más alto", xy=(0, 0.47), xytext=(1.3, 0.5), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7))
ax[1].annotate("hombros\nmás delgados", xy=(1.3, 0.13), xytext=(2.1, 0.27), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7))
ax[1].annotate("colas más\npesadas", xy=(2.9, 0.012), xytext=(3.0, 0.12), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7))
ax[1].set_title("Curtosis (misma media y varianza)"); ax[1].set_xlabel("$z$"); ax[1].legend(fontsize=7.5, loc="upper left")
ax[1].set_ylim(0, 0.62)
for a in ax: eg.coma(a)
fig.savefig("graficos/cap02_fig_momentos.pdf")

# --- Figura: t de Student frente a la normal, escala lineal y logarítmica ---
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
casos = [(3, eg.LADRILLO, "--"), (5, eg.OLIVA, "-."), (10, eg.AZUL, ":")]
for a in ax:
    a.plot(x, stats.norm.pdf(x), color="black", label="normal ($\\nu=\\infty$)")
    for nu, col, ls in casos:
        a.plot(x, stats.t.pdf(x, nu), color=col, ls=ls, label=f"$t_{{{nu}}}$")
ax[0].set_title("Densidades (escala lineal)"); ax[0].set_ylabel("densidad")
ax[1].set_yscale("log"); ax[1].set_ylim(1e-6, 1); ax[1].set_title("Densidades (escala logarítmica)")
ax[1].yaxis.set_major_formatter(plt.LogFormatterMathtext())
for a in ax: a.set_xlabel("$x$"); a.legend(fontsize=7.5, loc="upper left" if a is ax[0] else "lower center")
eg.coma(ax[0]); ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
fig.savefig("graficos/cap02_fig_tstudent.pdf")

# --- Figura: colas Pareto frente a normal y t (escala log-log) ---
fig, ax = plt.subplots(figsize=(6.2, 3.6))
u = np.logspace(0, 1.5, 200)
for alfa, col, ls in [(2, eg.LADRILLO, "-"), (3, eg.OLIVA, "-"), (4, eg.AZUL, "-")]:
    ax.plot(u, u**(-alfa), color=col, ls=ls, label=f"Pareto, $\\alpha={alfa}$: pendiente $-{alfa}$")
ax.plot(u, stats.t.sf(u, 3) / stats.t.sf(1, 3), color=eg.OLIVA, ls="--", label="$t_3$ (cola $\\approx$ Pareto, $\\alpha=3$)")
ax.plot(u, stats.norm.sf(u) / stats.norm.sf(1), color="black", ls=":", label="normal")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-6, 1.5)
ax.set_xlabel("umbral $x$ (escala log)"); ax.set_ylabel("$P(X>x)\\,/\\,P(X>1)$ (escala log)")
ax.legend(fontsize=7.5, loc="lower left")
fig.savefig("graficos/cap02_fig_pareto.pdf")

# --- Figura: ARCH(1) simulado frente a ruido i.i.d. con la misma varianza ---
from statsmodels.tsa.stattools import acf
rng = np.random.default_rng(7); T = 1500; w, a = 0.5, 0.5          # varianza no condicional w/(1-a) = 1
e = rng.standard_normal(T); r = np.zeros(T); r[0] = e[0]
for t in range(1, T):
    r[t] = np.sqrt(w + a * r[t - 1]**2) * e[t]
iid = rng.standard_normal(T) * np.sqrt(w / (1 - a))
fig, ax = plt.subplots(2, 3, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [2.2, 1, 1]})
for fila, (serie, nom) in enumerate([(iid, "i.i.d. normal"), (r, "ARCH(1), $a=0{,}5$")]):
    ax[fila, 0].plot(serie, color=eg.AZUL if fila == 0 else eg.LADRILLO, lw=0.5)
    ax[fila, 0].set_ylim(-6, 6); ax[fila, 0].set_ylabel(nom)
    for j, (s, tit) in enumerate([(serie, "ACF de $r_t$"), (serie**2, "ACF de $r_t^2$")], start=1):
        ax[fila, j].vlines(range(1, 21), 0, acf(s, nlags=20)[1:], color=eg.AZUL if fila == 0 else eg.LADRILLO)
        ax[fila, j].axhline(0, color="black", lw=0.6)
        for sg in (1, -1): ax[fila, j].axhline(sg * 1.96 / np.sqrt(T), color=eg.GRIS, ls="--", lw=0.8)
        ax[fila, j].set_ylim(-0.15, 0.6)
        if fila == 0: ax[fila, j].set_title(tit)
ax[0, 0].set_title("Serie simulada (misma varianza no condicional)")
ax[1, 0].set_xlabel("tiempo"); ax[1, 1].set_xlabel("rezago"); ax[1, 2].set_xlabel("rezago")
for a_ in ax.flat: eg.coma(a_)
fig.tight_layout(); fig.savefig("graficos/cap02_fig_arch.pdf")
