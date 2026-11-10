# Capítulo 12: gráficas de la teoría (sesgo de Forbes-Rigobon y curva de impacto de la correlación DCC)
import numpy as np
import estilo_graficos as eg
import matplotlib.pyplot as plt

# --- Correlación medida en un período de alta volatilidad, sin contagio (Forbes y Rigobon, 2002) ---
delta = np.linspace(0, 10, 300)
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
for rho, col, ls in [(0.1, eg.OLIVA, ":"), (0.3, eg.AZUL, "-"), (0.5, eg.LADRILLO, "--")]:
    ax[0].plot(delta, rho * np.sqrt(1 + delta) / np.sqrt(1 + delta * rho**2), color=col, ls=ls, lw=1.5,
               label=f"$\\rho={rho}$".replace(".", ","))
    print(f"rho = {rho}: con delta = 3 la correlación medida es {rho * 2 / np.sqrt(1 + 3 * rho**2):.4f}")
ax[0].set_xlabel("aumento relativo de la varianza de $x$, $\\delta$"); ax[0].set_ylabel("correlación en la crisis")
ax[0].set_title("Sin contagio: $\\beta$ y $\\mathrm{Var}(e)$ constantes", fontsize=9); ax[0].legend(fontsize=8); eg.coma(ax[0])

# --- Correlación de mañana según los choques de hoy (DCC con a = 0,05; b = 0,93; correlación de largo plazo 0,3) ---
a, b, rb = 0.05, 0.93, 0.3
z1, z2 = np.meshgrid(np.linspace(-3, 3, 241), np.linspace(-3, 3, 241))
q11 = (1 - a - b) + a * z1**2 + b; q22 = (1 - a - b) + a * z2**2 + b; q12 = (1 - a - b) * rb + a * z1 * z2 + b * rb
rho = q12 / np.sqrt(q11 * q22)
cs = ax[1].contourf(z1, z2, rho, levels=np.arange(0.0, 0.62, 0.04), cmap="Blues")
ax[1].contour(z1, z2, rho, levels=[0.3], colors=[eg.LADRILLO], linewidths=1.2)
cb = fig.colorbar(cs, ax=ax[1]); cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
ax[1].set_xlabel("choque estandarizado $z_{1,t-1}$"); ax[1].set_ylabel("$z_{2,t-1}$")
ax[1].set_title("DCC: $\\rho_t$ partiendo de $\\rho_{t-1}=0{,}3$", fontsize=9); ax[1].set_aspect("equal"); eg.coma(ax[1])
for zz in [(2, 2), (2, -2), (0, 0)]:
    q11_ = (1 - a - b) + a * zz[0]**2 + b; q22_ = (1 - a - b) + a * zz[1]**2 + b; q12_ = (1 - a - b) * rb + a * zz[0] * zz[1] + b * rb
    print(f"DCC: choques {zz}: correlación de mañana = {q12_ / np.sqrt(q11_ * q22_):.4f}")
fig.tight_layout(); fig.savefig("graficos/cap12_fig_teoria.pdf")
