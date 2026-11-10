# Capítulo 11: gráficas de la teoría (paseos aleatorios independientes frente a cointegrados; corrección de errores)
import numpy as np
import estilo_graficos as eg
import matplotlib.pyplot as plt

rng = np.random.default_rng(1111); T = 400
w1 = np.cumsum(rng.standard_normal(T)); w2 = np.cumsum(rng.standard_normal(T))        # independientes
tendencia = np.cumsum(rng.standard_normal(T))                                         # tendencia común
z = np.zeros(T); e = rng.standard_normal(T)
for t in range(1, T): z[t] = 0.7 * z[t - 1] + e[t]                                    # desvío estacionario
c1 = tendencia + 0.5 * z; c2 = tendencia - 0.5 * z                                    # c1 - c2 = z
fig, ax = plt.subplots(2, 2, figsize=(9.4, 5.2), sharex=True)
ax[0, 0].plot(w1, color=eg.AZUL, lw=1); ax[0, 0].plot(w2, color=eg.LADRILLO, lw=1, ls="--")
ax[0, 0].set_title("Dos paseos aleatorios independientes")
ax[1, 0].plot(w1 - w2, color=eg.GRIS, lw=1); ax[1, 0].set_title("Su diferencia: también es un paseo aleatorio")
ax[0, 1].plot(c1, color=eg.AZUL, lw=1); ax[0, 1].plot(c2, color=eg.LADRILLO, lw=1, ls="--")
ax[0, 1].set_title("Dos series cointegradas (tendencia común)")
ax[1, 1].plot(c1 - c2, color=eg.GRIS, lw=1); ax[1, 1].set_title("Su diferencia: estacionaria, vuelve a cero")
for a in ax.ravel(): a.axhline(0, color="black", lw=0.4); eg.coma(a)
ax[1, 0].set_xlabel("tiempo"); ax[1, 1].set_xlabel("tiempo")
fig.tight_layout(); fig.savefig("graficos/cap11_fig_coint_sim.pdf")

# --- Corrección de errores tras un choque de +1 en y1 (VECM sin rezagos ni constante) ---
beta = np.array([1.0, -1.0])
casos = [("$\\alpha=(-0{,}05;\\ 0{,}20)$: $y_2$ corrige", np.array([-0.05, 0.20])),
         ("$\\alpha=(-0{,}20;\\ 0{,}05)$: $y_1$ corrige", np.array([-0.20, 0.05]))]
H = 30; fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4), sharey=True)
for a, (tit, alpha) in zip(ax, casos):
    y = np.zeros((H + 1, 2)); y[0] = [1.0, 0.0]
    for t in range(1, H + 1): y[t] = y[t - 1] + alpha * (beta @ y[t - 1])
    lim = (alpha[1] * 1.0 - alpha[0] * 0.0) / (alpha[1] - alpha[0])   # nivel común final
    print(tit, "nivel final:", np.round(y[-1], 4), "nivel teórico:", round(alpha[1] / (alpha[1] - alpha[0]), 4),
          "persistencia 1 + beta'alpha =", round(1 + beta @ alpha, 2))
    a.plot(y[:, 0], color=eg.AZUL, lw=1.5, label="$y_1$"); a.plot(y[:, 1], color=eg.LADRILLO, lw=1.5, ls="--", label="$y_2$")
    a.plot(y[:, 0] - y[:, 1], color=eg.GRIS, lw=1, ls=":", label="desequilibrio $y_1-y_2$")
    a.axhline(0, color="black", lw=0.4); a.set_title(tit, fontsize=9); a.set_xlabel("períodos tras el choque"); eg.coma(a)
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig("graficos/cap11_fig_ajuste.pdf")
