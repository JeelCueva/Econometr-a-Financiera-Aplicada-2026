# Capítulo 15: gráficas de la teoría (cargas y formas de Nelson-Siegel; trayectorias de Vasicek)
import numpy as np
import estilo_graficos as eg
import matplotlib.pyplot as plt
from cap15_funciones import ns_cargas

tau = np.linspace(0.5, 120, 300); X = ns_cargas(tau, 0.0609)
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))
for j, (et, col, ls) in enumerate([("nivel ($\\beta_0$)", eg.AZUL, "-"), ("pendiente ($\\beta_1$)", eg.LADRILLO, "--"), ("curvatura ($\\beta_2$)", eg.OLIVA, ":")]):
    ax[0].plot(tau, X[:, j], color=col, ls=ls, lw=1.5, label=et)
from scipy.optimize import brentq                     # máximo exacto: derivada de la carga de curvatura igual a cero
x_ = brentq(lambda x: (np.exp(-x) * (x**2 + x + 1) - 1) / x**2, 0.5, 5); c_ = ns_cargas([x_ / 0.0609], 0.0609)[0, 2]
print(f"la carga de curvatura es máxima en lambda*tau = {x_:.4f}, es decir, tau = {x_ / 0.0609:.2f} meses (valor {c_:.4f})")
ax[0].set_title("Cargas de Nelson-Siegel ($\\lambda=0{,}0609$)", fontsize=9); ax[0].set_xlabel("vencimiento (meses)"); ax[0].legend(fontsize=8)
for (b, et, col, ls) in [((6, -2, 0), "normal: $\\beta=(6;\\ -2;\\ 0)$", eg.AZUL, "-"), ((5, 2, 0), "invertida: $\\beta=(5;\\ 2;\\ 0)$", eg.LADRILLO, "--"),
                         ((6, -1, 4), "con joroba: $\\beta=(6;\\ -1;\\ 4)$", eg.OLIVA, ":")]:
    ax[1].plot(tau, X @ np.array(b), color=col, ls=ls, lw=1.5, label=et)
ax[1].set_title("Formas de la curva", fontsize=9); ax[1].set_xlabel("vencimiento (meses)"); ax[1].set_ylabel("rendimiento (% anual)")
ax[1].legend(fontsize=8)
for a in ax: eg.coma(a)
fig.tight_layout(); fig.savefig("graficos/cap15_fig_nelson_siegel.pdf")

# --- Vasicek: dr = kappa (theta - r) dt + sigma dW, simulado de forma exacta ---
kappa, theta, sigma, D = 0.5, 5.0, 1.0, 1 / 12; rng = np.random.default_rng(15); n = 12 * 15
b = np.exp(-kappa * D); s = sigma * np.sqrt((1 - b**2) / (2 * kappa))
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [2, 1]})
t = np.arange(n + 1) * D
for r0, col in [(1.0, eg.AZUL), (5.0, eg.GRIS), (10.0, eg.LADRILLO)]:
    r = np.empty(n + 1); r[0] = r0
    for i in range(n): r[i + 1] = theta + b * (r[i] - theta) + s * rng.standard_normal()
    ax[0].plot(t, r, color=col, lw=0.9); ax[0].plot(t, theta + (r0 - theta) * np.exp(-kappa * t), color=col, lw=1.4, ls="--")
ax[0].axhline(theta, color="black", lw=0.6); ax[0].set_xlabel("años"); ax[0].set_ylabel("tasa corta (% anual)")
ax[0].set_title("Trayectorias y su media condicional (discontinua)", fontsize=9)
sd = sigma / np.sqrt(2 * kappa); x = np.linspace(theta - 4 * sd, theta + 4 * sd, 200)
ax[1].plot(np.exp(-(x - theta)**2 / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi)), x, color=eg.AZUL, lw=1.4)
ax[1].set_title("Distribución estacionaria", fontsize=9); ax[1].set_xlabel("densidad")
print(f"Vasicek: kappa = {kappa}, theta = {theta}, sigma = {sigma}; vida media = {np.log(2) / kappa:.3f} años; desv. est. estacionaria = {sd:.4f}")
for a in ax: eg.coma(a)
fig.tight_layout(); fig.savefig("graficos/cap15_fig_vasicek.pdf")
