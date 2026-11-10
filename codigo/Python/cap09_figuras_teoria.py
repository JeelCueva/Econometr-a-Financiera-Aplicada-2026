# Capítulo 9: gráficas de la teoría (apalancamiento, función g(z) del EGARCH y curvas de impacto de noticias)
import numpy as np
import estilo_graficos as eg
import matplotlib.pyplot as plt

# --- Efecto apalancamiento de Black: volatilidad de la acción cuando cae el valor de la empresa ---
V = np.linspace(60, 160, 400); D = 50.0; sV = 20.0
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
ax[0].plot(V - D, sV * V / (V - D), color=eg.LADRILLO, lw=1.4)
ax[0].axhline(sV, color="black", ls=":", lw=0.8)
ax[0].set_xlabel("valor de las acciones $E=V-D$"); ax[0].set_ylabel("volatilidad de las acciones $\\sigma_E$ (%)")
ax[0].set_title("Deuda $D=50$, volatilidad de los activos $\\sigma_V=20$ %")
for Vv, dx, dy in ((150, -22, 12), (100, 4, 10), (70, 5, 6)):
    E_ = int(Vv - D); sE = sV * Vv / (Vv - D)
    ax[0].plot(E_, sE, "o", color=eg.AZUL, ms=4)
    ax[0].annotate(f"$E={E_}$: {sE:.0f} %", (E_, sE), (E_ + dx, sE + dy), fontsize=8)
eg.coma(ax[0])
# --- g(z) del EGARCH ---
z = np.linspace(-4, 4, 400); c = np.sqrt(2 / np.pi)
for a, g, col, ls, et in [(0.13, 0.0, eg.GRIS, ":", "$\\alpha=0{,}13$, $\\gamma=0$ (simétrica)"),
                          (0.13, -0.15, eg.AZUL, "-", "$\\alpha=0{,}13$, $\\gamma=-0{,}15$")]:
    ax[1].plot(z, a * (np.abs(z) - c) + g * z, color=col, ls=ls, lw=1.4, label=et)
ax[1].axhline(0, color="black", lw=0.5); ax[1].axvline(0, color="black", lw=0.5)
ax[1].set_xlabel("$z_{t-1}$"); ax[1].set_ylabel("$g(z_{t-1})$"); ax[1].set_title("Función de noticias del EGARCH")
ax[1].legend(fontsize=8); eg.coma(ax[1])
fig.savefig("graficos/cap09_fig_apalancamiento.pdf")

# --- Curvas de impacto teóricas con la misma varianza incondicional (1) ---
x = np.linspace(-4, 4, 400); s2 = 1.0
garch = 0.02 + 0.10 * x**2 + 0.88 * s2                 # alpha + beta = 0,98
gjr = 0.02 + (0.02 + 0.16 * (x < 0)) * x**2 + 0.88 * s2  # alpha + gamma/2 + beta = 0,98
egarch = np.exp(0.0 + 0.13 * (np.abs(x) - c) - 0.15 * x + 0.98 * np.log(s2))
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(x, garch, color=eg.GRIS, ls=":", lw=1.5, label="GARCH: parábola simétrica")
ax.plot(x, gjr, color=eg.LADRILLO, lw=1.5, label="GJR: parábola con dos ramas")
ax.plot(x, egarch, color=eg.AZUL, ls="--", lw=1.5, label="EGARCH: exponencial asimétrica")
ax.axvline(0, color="black", lw=0.5); ax.set_ylim(0.5, 4)
ax.set_xlabel("choque $\\varepsilon_{t-1}$ (con $\\sigma_{t-1}^2=1$)"); ax.set_ylabel("$\\sigma_t^2$")
ax.legend(fontsize=8); eg.coma(ax); fig.savefig("graficos/cap09_fig_nic_teoria.pdf")
