# Capítulo 17: gráficas de la teoría (trayectoria simulada de un proceso con dos regímenes; mezcla frente a normal)
import numpy as np
from scipy.stats import norm
import estilo_graficos as eg
import matplotlib.pyplot as plt

mu, s, p11, p22 = np.array([1.0, -0.5]), np.array([3.5, 8.0]), 0.98, 0.90
pi1 = (1 - p22) / (2 - p11 - p22); pi = np.array([pi1, 1 - pi1])
m = pi @ mu; v = pi @ (s**2 + mu**2) - m**2
k4 = (pi @ (3 * s**4 + 6 * s**2 * (mu - m)**2 + (mu - m)**4)) / v**2
print(f"pi1 = {pi1:.4f}; duraciones: {1 / (1 - p11):.1f} y {1 / (1 - p22):.1f} meses; media = {m:.4f}; desv. est. = {np.sqrt(v):.4f}; curtosis = {k4:.4f}")
rng = np.random.default_rng(17); T = 300; st = np.zeros(T, int); y = np.zeros(T)
for t in range(T):
    if t: st[t] = st[t - 1] if rng.uniform() < (p11 if st[t - 1] == 0 else p22) else 1 - st[t - 1]
    y[t] = mu[st[t]] + s[st[t]] * rng.standard_normal()
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
for t in range(T):
    if st[t] == 1: ax[0].axvspan(t - 0.5, t + 0.5, color=eg.GRIS, alpha=0.3, lw=0)
ax[0].plot(y, color=eg.AZUL, lw=0.8); ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$y_t$")
ax[0].set_title("Simulación; en gris, el régimen 2 (volátil)", fontsize=9)
x = np.linspace(-30, 30, 600); fm = pi[0] * norm.pdf(x, mu[0], s[0]) + pi[1] * norm.pdf(x, mu[1], s[1])
ax[1].plot(x, fm, color=eg.AZUL, lw=1.5, label="mezcla de dos normales"); ax[1].plot(x, norm.pdf(x, m, np.sqrt(v)), color=eg.LADRILLO, ls="--", lw=1.3, label="normal con igual media y varianza")
ax[1].set_title("Distribución incondicional", fontsize=9); ax[1].set_xlabel("$y$"); ax[1].legend(fontsize=7)
for a in ax: eg.coma(a)
fig.tight_layout(); fig.savefig("graficos/cap17_fig_regimenes.pdf")
