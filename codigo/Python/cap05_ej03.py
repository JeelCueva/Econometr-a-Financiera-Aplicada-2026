# Capítulo 5, Ejercicio 5.3: la tasa de interés de largo plazo de EE. UU., ADF paso a paso
import numpy as np, pandas as pd
from cap05_funciones import adf, kpss

s = pd.read_csv("datos/internacional/sp500_mensual_shiller.csv", parse_dates=["Date"])
s = s[s.Date <= "2023-06-01"]
i = s["Long Interest Rate"].to_numpy()      # rendimiento del bono del Tesoro a 10 años (%)
print(f"T = {len(i)} meses ({s.Date.iloc[0].date()} a {s.Date.iloc[-1].date()})")
a = adf(i, "c")
print(f"Rezagos elegidos por AIC: p = {a['p']}")
nom = ["gamma (y_{t-1})"] + [f"delta_{j}" for j in range(1, a["p"] + 1)] + ["constante"]
tab = pd.DataFrame({"coef": a["coef"], "ee": a["ee"]}, index=nom).iloc[[0, 1, 2, a["p"] + 1]]
print("Regresión ADF (primeros coeficientes):"); print(tab.round(5).to_string())
print(f"t de gamma = {a['coef'][0]:.5f} / {a['ee'][0]:.5f} = {a['t']:.3f}  (VC: 1% -3,43 | 5% -2,86 | 10% -2,57)")
ad = adf(np.diff(i), "c")
print(f"En diferencias: t = {ad['t']:.3f} con {ad['p']} rezagos")
print(f"KPSS en niveles: eta = {kpss(i)['eta']:.3f} | en diferencias: eta = {kpss(np.diff(i))['eta']:.3f} (VC 5% = 0,463)")

# ---- Figura ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(8.6, 4.8), sharex=True)
ax[0].plot(s.Date, i, color=eg.LADRILLO, lw=0.8); ax[0].set_ylabel("% anual"); ax[0].set_title("Rendimiento del bono del Tesoro a 10 años")
ax[1].plot(s.Date.iloc[1:], np.diff(i), color=eg.AZUL, lw=0.5); ax[1].set_ylabel("puntos porcentuales"); ax[1].set_title("Cambio mensual")
for a_ in ax: a_.yaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: f"{v_:g}".replace(".", ",")))
fig.tight_layout(); fig.savefig("graficos/cap05_fig_tasa.pdf")
