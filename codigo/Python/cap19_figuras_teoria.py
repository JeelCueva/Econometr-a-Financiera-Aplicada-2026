# Capítulo 19: gráficas de la teoría (sesgo por heterogeneidad no observada; factor de Moulton)
import numpy as np
import estilo_graficos as eg
import matplotlib.pyplot as plt

rng = np.random.default_rng(19); fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6)); cols = [eg.AZUL, eg.LADRILLO, eg.OLIVA, eg.GRIS]
X, Y = [], []
for i, a in enumerate([0, 3, 6, 9]):
    x = 2 * i + rng.normal(0, 1, 25); y = a - 0.5 * x + rng.normal(0, 0.6, 25); X.append(x); Y.append(y)
    ax[0].plot(x, y, "o", ms=3, color=cols[i]); xs = np.linspace(x.min(), x.max(), 5); ax[0].plot(xs, y.mean() - 0.5 * (xs - x.mean()), color=cols[i], lw=1)
x, y = np.concatenate(X), np.concatenate(Y); b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 5)
ax[0].plot(xs, np.polyval(b, xs), color="black", ls="--", lw=1.3, label=f"MCO agrupado: pendiente {b[0]:.2f}".replace(".", ","))
ax[0].plot([], [], color=eg.GRIS, lw=1, label="dentro de cada unidad: pendiente $-0{,}5$"); ax[0].legend(fontsize=7, loc="upper left")
ax[0].set_xlabel("$x_{it}$"); ax[0].set_ylabel("$y_{it}$"); ax[0].set_title("Efecto individual correlacionado con $x$", fontsize=9)
m = np.arange(1, 101)
for r, ls in [(0.1, "-"), (0.3, "--"), (0.7, ":")]:
    ax[1].plot(m, np.sqrt(1 + (m - 1) * r * 1.0), color=eg.AZUL, ls=ls, lw=1.4, label=f"$\\rho_x\\rho_\\varepsilon={r}$".replace(".", "{,}"))
ax[1].set_xlabel("tamaño del grupo $m$"); ax[1].set_ylabel("ee agrupado / ee clásico"); ax[1].set_title("Factor de Moulton $\\sqrt{1+(m-1)\\rho_x\\rho_\\varepsilon}$", fontsize=9); ax[1].legend(fontsize=8)
for a in ax: eg.coma(a)
fig.tight_layout(); fig.savefig("graficos/cap19_fig_teoria.pdf")
print(f"pendiente agrupada en la simulación = {b[0]:.4f}; factor de Moulton con m = 30 y rho = 0,7 x 0,7: {np.sqrt(1 + 29 * 0.49):.4f}")
