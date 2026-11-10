# Capítulo 10: gráficas de la teoría (autovalores y forma de las respuestas; efecto del orden de Cholesky)
import numpy as np
import estilo_graficos as eg
import matplotlib.pyplot as plt
from cap10_funciones import irf

# --- Autovalores de la matriz compañera y forma de la respuesta ---
th = np.pi / 6
casos = [("autovalores reales 0,9 y 0,5", np.array([[0.9, 0.0], [0.2, 0.5]]), eg.AZUL, "-"),
         ("complejos $0{,}8\\,e^{\\pm i\\pi/6}$", 0.8 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]), eg.OLIVA, "--"),
         ("autovalor 1,02 (inestable)", np.array([[1.02, 0.0], [0.1, 0.5]]), eg.LADRILLO, ":")]
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
u = np.linspace(0, 2 * np.pi, 300); ax[0].plot(np.cos(u), np.sin(u), color="black", lw=0.8)
for et, A, col, ls in casos:
    ev = np.linalg.eigvals(A); print(et, np.round(ev, 4), "módulos", np.round(np.abs(ev), 4))
    ax[0].plot(ev.real, ev.imag, "o", color=col, ms=6, label=et)
    ax[1].plot(np.arange(31), irf([A], 30)[:, 0, 0], color=col, ls=ls, lw=1.5, label=et)
ax[0].axhline(0, color=eg.GRIS, lw=0.5); ax[0].axvline(0, color=eg.GRIS, lw=0.5)
ax[0].set_aspect("equal"); ax[0].set_xlim(-1.2, 1.2); ax[0].set_ylim(-1.2, 1.2)
ax[0].set_xlabel("parte real"); ax[0].set_ylabel("parte imaginaria"); ax[0].set_title("Círculo unitario")
ax[1].axhline(0, color="black", lw=0.5); ax[1].set_ylim(-0.6, 1.9)
ax[1].set_xlabel("horizonte $i$"); ax[1].set_ylabel("$\\Psi_i[1,1]$"); ax[1].set_title("Respuesta de $y_1$ a su propio choque")
ax[1].legend(fontsize=8)
for a in ax: eg.coma(a)
fig.tight_layout(); fig.savefig("graficos/cap10_fig_autovalores.pdf")

# --- Efecto del orden de Cholesky con errores correlacionados ---
A = np.array([[0.5, 0.1], [0.3, 0.6]]); S = np.array([[1.0, 0.6], [0.6, 1.0]])
P1 = np.linalg.cholesky(S)                                 # orden (y1, y2)
Pr = np.linalg.cholesky(S[::-1, ::-1])[::-1, ::-1]         # orden (y2, y1), reexpresado en (y1, y2)
h = 12; t1 = irf([A], h, P1); t2 = irf([A], h, Pr)
print("P orden (y1, y2):\n", np.round(P1, 4)); print("P orden (y2, y1):\n", np.round(Pr, 4))
fig, ax = plt.subplots(2, 2, figsize=(9.0, 5.2), sharex=True)
for k in range(2):
    for j in range(2):
        a = ax[k, j]
        a.plot(t1[:, k, j], color=eg.AZUL, lw=1.5, label="orden $(y_1, y_2)$")
        a.plot(t2[:, k, j], color=eg.LADRILLO, lw=1.5, ls="--", label="orden $(y_2, y_1)$")
        a.axhline(0, color="black", lw=0.5); a.set_title(f"respuesta de $y_{k + 1}$ al choque en $y_{j + 1}$"); eg.coma(a)
ax[1, 0].set_xlabel("horizonte"); ax[1, 1].set_xlabel("horizonte"); ax[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig("graficos/cap10_fig_orden.pdf")
