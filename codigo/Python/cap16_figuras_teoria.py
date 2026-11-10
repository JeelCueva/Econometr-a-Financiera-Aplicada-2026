# Capítulo 16: gráficas de la teoría (línea de tiempo; efecto de la varianza inducida por el evento)
import numpy as np
from scipy.stats import norm
import estilo_graficos as eg
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.6, 1.9))
ax.axhline(0, color="black", lw=0.8)
for (a, b, c, t) in [(-250, -11, eg.AZUL, "ventana de estimación\n$\\tau=-250,\\dots,-11$  ($L_1=240$)"),
                     (-5, 5, eg.LADRILLO, "ventana del evento\n$\\tau=-5,\\dots,+5$  ($L_2=11$)")]:
    ax.add_patch(plt.Rectangle((a, -0.18), b - a, 0.36, color=c, alpha=0.25)); ax.text((a + b) / 2, 0.32, t, ha="center", fontsize=8)
for x, t, ha, y in [(-250, "$T_0$", "center", -0.42), (-11, "$T_1$", "right", -0.42), (0, "día 0", "center", -0.72), (5, "$T_2$", "left", -0.42)]:
    ax.plot([x, x], [-0.22, 0.22], color="black", lw=0.8); ax.text(x, y, t, ha=ha, fontsize=8)
ax.annotate("", xy=(0, -0.24), xytext=(0, -0.58), arrowprops=dict(arrowstyle="->", lw=0.6))
ax.set_xlim(-262, 30); ax.set_ylim(-0.85, 0.8); ax.axis("off"); fig.savefig("graficos/cap16_fig_linea_tiempo.pdf")

x = np.linspace(-5, 5, 400); fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(x, norm.pdf(x), color=eg.AZUL, lw=1.5, label="distribución supuesta: $N(0,1)$")
ax.plot(x, norm.pdf(x, scale=1.5), color=eg.LADRILLO, lw=1.5, ls="--", label="verdadera, varianza $\\times2{,}25$: $N(0;\\,1{,}5^2)$")
for lado in (-1, 1):
    xx = np.linspace(1.96, 5, 100) * lado; ax.fill_between(xx, norm.pdf(xx, scale=1.5), color=eg.LADRILLO, alpha=0.2)
ax.axvline(1.96, color=eg.GRIS, ls=":", lw=1); ax.axvline(-1.96, color=eg.GRIS, ls=":", lw=1)
p = 2 * norm.sf(1.96 / 1.5); ax.set_xlabel("estadístico estandarizado"); ax.set_ylabel("densidad")
ax.legend(fontsize=8); eg.coma(ax); fig.savefig("graficos/cap16_fig_varianza_inducida.pdf")
print(f"Con varianza 1,5^2 y valor crítico 1,96, la probabilidad de rechazo bajo H0 es {p:.4f}")
