# Capítulo 8: gráficas de la teoría (ARCH simulado, región de momentos, pronósticos y t de Student)
import numpy as np
import estilo_graficos as eg
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(808)
# --- ARCH(1) frente a ruido normal con la misma varianza incondicional ---
def arch1(T, om, a, rng, quema=500):
    z = rng.standard_normal(T + quema); e = np.zeros(T + quema)
    for t in range(1, T + quema):
        e[t] = np.sqrt(om + a * e[t - 1]**2) * z[t]
    return e[quema:]
om, a = 0.5, 0.5                                   # varianza incondicional om/(1-a) = 1
x = arch1(1000, om, a, rng); n = rng.standard_normal(1000)
grande = arch1(400_000, om, a, rng)
k = ((grande - grande.mean())**4).mean() / grande.var()**2
print(f"ARCH(1) con omega = {om}, alpha = {a}: varianza simulada = {grande.var():.3f} (teórica 1)")
print(f"curtosis simulada = {k:.2f}; teórica 3(1-a^2)/(1-3a^2) = {3 * (1 - a**2) / (1 - 3 * a**2):.2f}")
fig = plt.figure(figsize=(9.4, 4.6)); g = fig.add_gridspec(2, 2, width_ratios=[1.7, 1])
a1 = fig.add_subplot(g[0, 0]); a2 = fig.add_subplot(g[1, 0], sharey=a1); a3 = fig.add_subplot(g[:, 1])
a1.plot(n, color=eg.AZUL, lw=0.6); a1.set_title("Ruido normal i.i.d., varianza 1")
a2.plot(x, color=eg.LADRILLO, lw=0.6); a2.set_title("ARCH(1): $\\omega=0{,}5$, $\\alpha=0{,}5$, varianza 1")
a2.set_xlabel("tiempo")
v = np.linspace(-8, 8, 400); bordes = np.arange(-8, 8.01, 0.25)
dens, _ = np.histogram(grande, bins=bordes, density=True); centros = (bordes[1:] + bordes[:-1]) / 2
a3.semilogy(v, stats.norm.pdf(v), color=eg.AZUL, ls="--", label="normal")
a3.semilogy(centros[dens > 0], dens[dens > 0], color=eg.LADRILLO, label="ARCH(1) (histograma)")
a3.set_ylim(1e-6, 1); a3.set_title("Densidades (escala log)"); a3.legend(fontsize=8); a3.set_xlabel("valor")
for ax in (a1, a2, a3): eg.coma(ax)
a3.yaxis.set_major_formatter(plt.LogFormatterMathtext())
fig.tight_layout(); fig.savefig("graficos/cap08_fig_arch_sim.pdf")

# --- Región de estacionariedad y de cuarto momento finito del GARCH(1,1) ---
al = np.linspace(0, 0.6, 600)
b_est = 1 - al                                         # alpha + beta = 1
b_k = -al + np.sqrt(np.clip(1 - 2 * al**2, 0, None))   # (alpha+beta)^2 + 2 alpha^2 = 1
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.9))
for ax, zoom in zip(axs, (False, True)):
    ax.fill_between(al, 0, np.clip(b_k, 0, 1), color=eg.AZUL, alpha=0.25, label="cuarto momento finito")
    ax.fill_between(al, np.clip(b_k, 0, 1), np.clip(b_est, 0, 1), color=eg.OLIVA, alpha=0.18,
                    label="estacionario, curtosis infinita")
    ax.plot(al, b_est, color="black", lw=0.9); ax.plot(al, b_k, color=eg.AZUL, lw=0.9)
    ax.plot(0.0902, 0.9074, "o", color=eg.LADRILLO, ms=5)
    ax.set_xlabel("$\\alpha$"); ax.set_ylabel("$\\beta$"); eg.coma(ax)
axs[0].set_xlim(0, 0.6); axs[0].set_ylim(0, 1.0); axs[0].legend(fontsize=8, loc="lower left")
axs[0].text(0.36, 0.69, "$\\alpha+\\beta=1$", fontsize=9)
axs[0].add_patch(plt.Rectangle((0.05, 0.85), 0.08, 0.10, fill=False, lw=0.8, ls=":"))
axs[0].set_title("Plano $(\\alpha,\\beta)$")
axs[1].set_xlim(0.05, 0.13); axs[1].set_ylim(0.85, 0.95); axs[1].set_title("Ampliación del recuadro")
axs[1].annotate("Brent (Ejercicio 8.2)", (0.0902, 0.9074), (0.095, 0.935), fontsize=8, color=eg.LADRILLO,
                arrowprops=dict(arrowstyle="-", color=eg.LADRILLO, lw=0.6))
fig.savefig("graficos/cap08_fig_region.pdf")

# --- Pronóstico de la varianza: convergencia geométrica a la varianza incondicional ---
h = np.arange(1, 251)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for p, col, ls in [(0.90, eg.OLIVA, ":"), (0.97, eg.AZUL, "--"), (0.995, eg.LADRILLO, "-")]:
    for s0 in (3.0, 0.4):
        ax.plot(h, 1 + p**(h - 1) * (s0 - 1), color=col, ls=ls, lw=1.1,
                label=f"$\\alpha+\\beta={p}$".replace(".", ",") if s0 == 3 else None)
ax.axhline(1, color="black", lw=0.6)
ax.set_xlabel("horizonte $h$"); ax.set_ylabel("$\\mathrm{E}_T[\\sigma^2_{T+h}]\\,/\\,\\bar\\sigma^2$")
ax.legend(fontsize=8); eg.coma(ax); fig.savefig("graficos/cap08_fig_pronostico_teo.pdf")

# --- t de Student estandarizada frente a la normal ---
def t_est(v, nu): return stats.t.pdf(v * np.sqrt(nu / (nu - 2)), nu) * np.sqrt(nu / (nu - 2))
v = np.linspace(-7, 7, 500)
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.3))
for axx, logy in zip(ax, (False, True)):
    f = axx.semilogy if logy else axx.plot
    f(v, stats.norm.pdf(v), color=eg.AZUL, ls="--", label="normal")
    f(v, t_est(v, 8), color=eg.OLIVA, label="$t$ estandarizada, $\\nu=8$")
    f(v, t_est(v, 4), color=eg.LADRILLO, label="$t$ estandarizada, $\\nu=4$")
    axx.set_xlabel("$z$"); eg.coma(axx)
ax[0].set_title("Escala lineal: más pico en el centro"); ax[1].set_title("Escala log: colas más gruesas")
ax[1].set_ylim(1e-7, 1); ax[1].yaxis.set_major_formatter(plt.LogFormatterMathtext()); ax[0].legend(fontsize=8)
for nu in (4, 8):
    print(f"t estandarizada nu = {nu}: P(|z| > 4) = {2 * stats.t.sf(4 * np.sqrt(nu / (nu - 2)), nu):.2e}")
print(f"normal: P(|z| > 4) = {2 * stats.norm.sf(4):.2e}")
fig.savefig("graficos/cap08_fig_t_student.pdf")
