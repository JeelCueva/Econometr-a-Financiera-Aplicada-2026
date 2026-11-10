# Capítulo 5, Ejercicio 5.2: el VIX, una serie estacionaria pero persistente
import numpy as np, pandas as pd
from cap05_funciones import adf, kpss

v = pd.read_csv("datos/internacional/vix_diario.csv", parse_dates=["DATE"])
y = np.log(v["CLOSE"].to_numpy()); T = len(y)
a, k = adf(y, "c"), kpss(y, "c")
print(f"T = {T} días ({v.DATE.iloc[0].date()} a {v.DATE.iloc[-1].date()})")
print(f"ADF: t = {a['t']:.3f} con {a['p']} rezagos (VC 5% = -2,86)")
print(f"KPSS: eta = {k['eta']:.3f} con l = {k['l']} (VC 5% = 0,463)")
X = np.column_stack([np.ones(T - 1), y[:-1]]); c, phi = np.linalg.lstsq(X, y[1:], rcond=None)[0]
mu = c / (1 - phi)
print(f"AR(1): phi = {phi:.4f}  media de largo plazo = {mu:.3f} (VIX = {np.exp(mu):.2f})")
print(f"Vida media de un choque = ln(0,5)/ln(phi) = {np.log(0.5)/np.log(phi):.1f} días")

# ---- Figura ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
ax[0].plot(v.DATE, y, color=eg.AZUL, lw=0.5); ax[0].axhline(mu, color=eg.LADRILLO, ls="--", lw=1, label="media de largo plazo")
ax[0].set_title("Logaritmo del VIX, 1990-2026"); ax[0].legend(fontsize=8)
j = np.arange(1, 251); ax[1].vlines(j, 0, acf(y, nlags=250)[1:], color=eg.AZUL, lw=0.6)
ax[1].plot(j, phi**j, color=eg.LADRILLO, ls="--", lw=1.2, label=r"$\hat\phi^{\,j}$ del AR(1)")
ax[1].set_title("ACF"); ax[1].set_xlabel("rezago (días)"); ax[1].legend(fontsize=8)
for a_ in ax: a_.yaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: f"{v_:g}".replace(".", ",")))
fig.savefig("graficos/cap05_fig_vix.pdf")
