# Capítulo 6: gráficas de la teoría (ACF y PACF teóricas, sesgo de Kendall, pronóstico de un AR(1))
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
import estilo_graficos as eg
import matplotlib.pyplot as plt

# --- Figura: ACF y PACF teóricas de cuatro modelos ---
modelos = [("AR(1): $\\phi=0{,}8$", [1, -0.8], [1]),
           ("AR(2): $\\phi_1=1{,}2$, $\\phi_2=-0{,}5$", [1, -1.2, 0.5], [1]),
           ("MA(1): $\\theta=0{,}6$", [1], [1, 0.6]),
           ("ARMA(1,1): $\\phi=0{,}8$, $\\theta=0{,}4$", [1, -0.8], [1, 0.4])]
fig, ax = plt.subplots(4, 2, figsize=(8.6, 7.4), sharex=True)
j = np.arange(1, 13)
for i, (nom, ar, ma) in enumerate(modelos):
    p = ArmaProcess(ar, ma)
    ax[i, 0].vlines(j, 0, p.acf(13)[1:], color=eg.AZUL, lw=2); ax[i, 1].vlines(j, 0, p.pacf(13)[1:], color=eg.LADRILLO, lw=2)
    ax[i, 0].set_ylabel(nom, fontsize=8)
    for a in ax[i]: a.axhline(0, color="black", lw=0.5); a.set_ylim(-0.75, 1.0); eg.coma(a)
ax[0, 0].set_title("ACF teórica"); ax[0, 1].set_title("PACF teórica")
ax[3, 0].set_xlabel("rezago $j$"); ax[3, 1].set_xlabel("rezago $j$")
fig.tight_layout(); fig.savefig("graficos/cap06_fig_identificacion.pdf")

# --- Figura: sesgo de Kendall del estimador MCO de phi ---
rng = np.random.default_rng(66); T, reps = 100, 4000
phis = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]); sesgo = []
for phi in phis:
    est = np.empty(reps)
    for r in range(reps):
        e = rng.standard_normal(T + 100); y = np.zeros(T + 100)
        for t in range(1, T + 100): y[t] = phi * y[t - 1] + e[t]
        y = y[100:]; x = y[:-1] - y[:-1].mean(); z = y[1:] - y[1:].mean(); est[r] = x @ z / (x @ x)
    sesgo.append(est.mean() - phi)
print("Sesgo simulado de phi (T = 100):", dict(zip(phis.round(2), np.round(sesgo, 4))))
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(phis, sesgo, "o-", color=eg.AZUL, label="simulación (4 000 réplicas)")
g = np.linspace(0, 0.99, 100); ax.plot(g, -(1 + 3 * g) / T, color=eg.LADRILLO, ls="--", label="fórmula de Kendall $-(1+3\\phi)/T$")
ax.axhline(0, color="black", lw=0.5); ax.set_xlabel("$\\phi$ verdadero"); ax.set_ylabel("$E[\\hat\\phi]-\\phi$")
ax.set_title("Sesgo del estimador MCO en un AR(1) con $T=100$"); ax.legend(fontsize=8); eg.coma(ax)
fig.savefig("graficos/cap06_fig_kendall.pdf")

# --- Figura: pronóstico de un AR(1) y su intervalo ---
fig, ax = plt.subplots(figsize=(7, 3.3))
mu, s2, yT = 0.0, 1.0, 3.0; h = np.arange(0, 41)
for phi, col, ls in [(0.5, eg.OLIVA, "-"), (0.9, eg.AZUL, "-"), (0.98, eg.LADRILLO, "--")]:
    f = mu + phi**h * (yT - mu); ee = np.sqrt(s2 * (1 - phi**(2 * h)) / (1 - phi**2))
    ax.plot(h, f, color=col, ls=ls, label=f"$\\phi={phi}$".replace(".", ","))
    ax.fill_between(h, f - 1.96 * ee, f + 1.96 * ee, color=col, alpha=0.10)
ax.axhline(mu, color="black", lw=0.5)
ax.set_xlabel("horizonte $h$"); ax.set_ylabel("$\\hat y_{T+h}$ e intervalo al 95 %"); ax.legend(fontsize=8)
ax.set_title("Pronóstico desde $y_T=3$ hacia la media $\\mu=0$"); eg.coma(ax)
fig.savefig("graficos/cap06_fig_pronostico_ar1.pdf")
