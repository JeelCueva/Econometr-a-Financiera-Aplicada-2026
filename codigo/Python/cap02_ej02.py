# Capítulo 2, Ejercicio 2.2: autocorrelación de r, r^2 y |r| (Brent diario)
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

datos = pd.read_csv("datos/internacional/brent_diario.csv")
r = np.log(datos["Price"]).diff().dropna().to_numpy()
T = len(r)
series = {"r": r, "r2": r**2, "abs_r": np.abs(r)}

print(f"Banda de Bartlett al 95 %: +/- {1.96/np.sqrt(T):.4f}\n")
tabla = pd.DataFrame({k: acf(v, nlags=10)[1:] for k, v in series.items()},
                     index=[f"rezago_{j}" for j in range(1, 11)])
print(tabla.round(4).to_string())

print("\nPrueba de Ljung-Box (H0: sin autocorrelación hasta el rezago m)")
for nombre, x in series.items():
    lb = acorr_ljungbox(x, lags=[10, 20])
    for m, fila in lb.iterrows():
        print(f"{nombre:<6} m = {m:2d}  Q = {fila.lb_stat:10.2f}  p-valor = {fila.lb_pvalue:.4g}")

# ---- Figura: ACF hasta el rezago 100 ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(9.5, 3), sharey=True)
titulos = {"r": "$r_t$", "r2": "$r_t^2$", "abs_r": "$|r_t|$"}
for a, (k, v) in zip(ax, series.items()):
    rho = acf(v, nlags=100)[1:]
    a.vlines(range(1, 101), 0, rho, color=eg.AZUL, lw=1)
    a.axhline(0, color="black", lw=0.6)
    for s in (1, -1): a.axhline(s * 1.96 / np.sqrt(T), color=eg.LADRILLO, ls="--", lw=0.9)
    a.set_title(titulos[k]); a.set_xlabel("rezago (días)"); eg.coma(a)
ax[0].set_ylabel("autocorrelación"); ax[0].set_ylim(-0.15, 0.35)
fig.savefig("graficos/cap02_fig_acf.pdf")
