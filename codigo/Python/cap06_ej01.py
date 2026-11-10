# Capítulo 6, Ejercicio 6.1: identificación de un ARMA para los cambios de la tasa a 10 años
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA

s = pd.read_csv("datos/internacional/sp500_mensual_shiller.csv", parse_dates=["Date"])
s = s[(s.Date >= "1953-01-01") & (s.Date <= "2023-06-01")]
di = np.diff(s["Long Interest Rate"].to_numpy()); T = len(di)
print(f"T = {T} cambios mensuales; banda 95 % = +/- {1.96/np.sqrt(T):.3f}")
print("ACF :", " ".join(f"{v:6.3f}" for v in acf(di, nlags=6)[1:]))
print("PACF:", " ".join(f"{v:6.3f}" for v in pacf(di, nlags=6, method="ywm")[1:]), "\n")
ordenes = {"AR(1)": (1, 0, 0), "AR(2)": (2, 0, 0), "MA(1)": (0, 0, 1), "MA(2)": (0, 0, 2), "ARMA(1,1)": (1, 0, 1)}
for n, o in ordenes.items():
    m = ARIMA(di, order=o, trend="c").fit()
    cf = {k: v for k, v in zip(m.model.param_names, m.params) if k.startswith(("ar", "ma"))}
    print(f"{n:<10} {' '.join(f'{k}={v:.4f}' for k, v in cf.items()):<32} AIC = {m.aic:7.2f}  BIC = {m.bic:7.2f}")
print(f"\nMA(1) implicado por el efecto de Working (rho1 = 1/4): theta = 2 - sqrt(3) = {2-np.sqrt(3):.4f}")

# ---- Figura: ACF y PACF muestrales ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
j = np.arange(1, 25)
for a, v, t in [(ax[0], acf(di, nlags=24)[1:], "ACF muestral"), (ax[1], pacf(di, nlags=24, method="ywm")[1:], "PACF muestral")]:
    a.vlines(j, 0, v, color=eg.AZUL, lw=2); a.axhline(0, color="black", lw=0.5)
    for sg in (1, -1): a.axhline(sg * 1.96 / np.sqrt(T), color=eg.LADRILLO, ls="--", lw=0.8)
    a.set_title(t); a.set_xlabel("rezago (meses)"); eg.coma(a)
fig.savefig("graficos/cap06_fig_tasa_acf.pdf")
