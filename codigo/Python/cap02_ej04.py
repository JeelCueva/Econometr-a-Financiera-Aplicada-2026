# Capítulo 2, Ejercicio 2.4: el S&P 500 mensual de Shiller y el efecto del promedio (Working, 1960)
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

datos = pd.read_csv("datos/internacional/sp500_mensual_shiller.csv", parse_dates=["Date"])
r = np.log(datos.set_index("Date")["SP500"]).diff().dropna()
print(f"Período: {r.index.min().date()} a {r.index.max().date()}  T = {len(r)}\n")

tramos = {"completo": r, "1871-1945": r[:"1945-12-01"], "1946-2026": r["1946-01-01":]}
for n, x in tramos.items():
    rho = acf(x, nlags=6)[1:]
    print(f"{n:<10} T = {len(x):4d}  banda = {1.96/np.sqrt(len(x)):.3f}  rho(1..6) =",
          " ".join(f"{v:6.3f}" for v in rho))
print(f"\nValor teórico de Working para promedios de 21 días: {(21**2-1)/(2*(2*21**2+1)):.4f}")
lb = acorr_ljungbox(r, lags=[1])
print(f"Ljung-Box (m = 1): Q = {lb.lb_stat.iloc[0]:.2f}, p-valor = {lb.lb_pvalue.iloc[0]:.3g}")

# ---- Figura: simulación del efecto Working frente a los datos ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
rng = np.random.default_rng(2026)
diario = np.cumsum(rng.standard_normal(21 * 1868))          # paseo aleatorio, 21 días por mes
fin_mes = diario[20::21]; promedio = diario.reshape(-1, 21).mean(axis=1)
fig, ax = plt.subplots(figsize=(6.2, 3.3))
k = np.arange(1, 7); w = 0.27
ax.bar(k - w, acf(np.diff(fin_mes), nlags=6)[1:], w, color=eg.GRIS, label="simulado: precio de fin de mes")
ax.bar(k, acf(np.diff(promedio), nlags=6)[1:], w, color=eg.OLIVA, label="simulado: promedio del mes")
ax.bar(k + w, acf(r, nlags=6)[1:], w, color=eg.AZUL, label="S&P 500 (Shiller, promedio del mes)")
ax.axhline(0.25, color=eg.LADRILLO, ls="--", lw=1, label="valor teórico 1/4")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("rezago (meses)"); ax.set_ylabel("autocorrelación"); ax.legend(fontsize=7.5); eg.coma(ax)
fig.savefig("graficos/cap02_fig_working.pdf")
