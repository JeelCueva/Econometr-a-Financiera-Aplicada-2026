# Capítulo 3, Ejercicio 3.5: el portafolio tangente de las 12 industrias y la crítica de Roll
import numpy as np, pandas as pd

d = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv")
industrias = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
              "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]
Z = d[industrias].sub(d["RF"], axis=0)      # excesos de rendimiento
mu = Z.mean().to_numpy(); S = np.cov(Z.to_numpy(), rowvar=False)
w = np.linalg.solve(S, mu); w = w / w.sum() # pesos del portafolio tangente
zq = Z.to_numpy() @ w
print("Pesos del portafolio tangente (%):"); print(pd.Series(100 * w, index=industrias).round(1).to_string())
sharpe = lambda z: np.sqrt(12) * np.mean(z) / np.std(z, ddof=1)
print(f"\nSharpe anual: tangente {sharpe(zq):.3f} | mercado {sharpe(d['MktRF']):.3f}")

b_q = np.array([np.cov(Z[i], zq)[0, 1] for i in industrias]) / np.var(zq, ddof=1)
print(f"Máximo |mu_i - beta_iq * mu_q| = {np.max(np.abs(mu - b_q * zq.mean())):.2e}")
xm = d["MktRF"]
b_m = np.array([np.cov(Z[i], xm)[0, 1] for i in industrias]) / xm.var()
alfa_m = mu - b_m * xm.mean()
print(f"Máximo |alfa_i| respecto del mercado = {1200 * np.max(np.abs(alfa_m)):.2f} % anual")

# ---- Figura: frontera eficiente, recta del mercado de capitales y activos ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
Si = np.linalg.inv(S); uno = np.ones(len(mu))
A, B, C = uno @ Si @ uno, uno @ Si @ mu, mu @ Si @ mu
m = np.linspace(-0.002, 0.016, 300)                          # medias mensuales objetivo
sd_front = np.sqrt((A * m**2 - 2 * B * m + C) / (A * C - B**2))
an = lambda media, desv: (1200 * media, 100 * np.sqrt(12) * desv)
fig, ax = plt.subplots(figsize=(6.2, 4.2))
y_f, x_f = an(m, sd_front); ax.plot(x_f, y_f, color=eg.AZUL, label="frontera de las 12 industrias")
sr = np.sqrt(C)                                              # Sharpe mensual del tangente
xs = np.linspace(0, 30, 2); ax.plot(xs, xs * sr * np.sqrt(12), color=eg.LADRILLO, ls="--", label="recta del mercado de capitales")
y_i, x_i = an(mu, np.sqrt(np.diag(S))); ax.scatter(x_i, y_i, s=14, color=eg.GRIS, zorder=3, label="industrias")
y_q, x_q = an(zq.mean(), zq.std(ddof=1)); ax.scatter(x_q, y_q, marker="*", s=120, color=eg.LADRILLO, zorder=4, label="tangente")
y_m, x_m = an(xm.mean(), xm.std()); ax.scatter(x_m, y_m, marker="D", s=30, color=eg.OLIVA, zorder=4, label="mercado")
ax.set_xlim(0, 30); ax.set_ylim(-2, 16)
ax.set_xlabel("volatilidad (% anual)"); ax.set_ylabel("exceso de rendimiento medio (% anual)")
ax.legend(fontsize=8, loc="upper left"); eg.coma(ax)
fig.savefig("graficos/cap03_fig_frontera.pdf")
