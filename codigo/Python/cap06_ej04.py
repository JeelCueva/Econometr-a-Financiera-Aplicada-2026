# Capítulo 6, Ejercicio 6.4: pronóstico del VIX hacia adelante con intervalos de confianza
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA

v = pd.read_csv("datos/internacional/vix_diario.csv"); y = np.log(v["CLOSE"].to_numpy()); T = len(y)
m = ARIMA(y, order=(1, 0, 1), trend="c").fit(); mu, phi, th, s2 = m.params
e = np.zeros(T)
for t in range(1, T): e[t] = (y[t] - mu) - phi * (y[t - 1] - mu) - th * e[t - 1]
H = 60; h = np.arange(1, H + 1)
f1 = mu + phi * (y[-1] - mu) + th * e[-1]
f = mu + phi**(h - 1) * (f1 - mu)
psi = np.r_[1, (phi + th) * phi**np.arange(H - 1)]           # pesos psi del MA(infinito)
ecm = s2 * np.cumsum(psi**2)                                  # error cuadrático medio a h pasos
print(f"Origen: {v.DATE.iloc[-1]}, VIX = {np.exp(y[-1]):.2f}")
print("   h   mediana   IC 95 %")
for k in (1, 5, 22, 60):
    print(f"{k:4d} {np.exp(f[k-1]):9.2f}   [{np.exp(f[k-1]-1.96*np.sqrt(ecm[k-1])):5.2f}; {np.exp(f[k-1]+1.96*np.sqrt(ecm[k-1])):5.2f}]")
var_lp = s2 * (1 + (phi + th)**2 / (1 - phi**2))
print(f"Límite de largo plazo: mediana {np.exp(mu):.2f}, IC 95 % [{np.exp(mu-1.96*np.sqrt(var_lp)):.2f}; {np.exp(mu+1.96*np.sqrt(var_lp)):.2f}]")
print("\nComprobación con statsmodels (h = 22):", np.round(np.exp(m.get_forecast(22).predicted_mean[-1]), 2))

# ---- Figura: abanico de pronóstico ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fechas = pd.to_datetime(v.DATE); fut = pd.bdate_range(fechas.iloc[-1], periods=H + 1)[1:]
fig, ax = plt.subplots(figsize=(8.4, 3.4))
ax.plot(fechas.iloc[-250:], np.exp(y[-250:]), color="black", lw=0.8, label="VIX observado")
for z, al in [(1.96, 0.15), (1.0, 0.30)]:
    ax.fill_between(fut, np.exp(f - z * np.sqrt(ecm)), np.exp(f + z * np.sqrt(ecm)), color=eg.AZUL, alpha=al, lw=0)
ax.plot(fut, np.exp(f), color=eg.AZUL, lw=1.2, label="mediana del pronóstico (bandas: 68 % y 95 %)")
ax.axhline(np.exp(mu), color=eg.LADRILLO, ls="--", lw=1, label="nivel de largo plazo")
ax.set_ylabel("VIX"); ax.legend(fontsize=8, loc="upper left")
fig.savefig("graficos/cap06_fig_vix_abanico.pdf")
