# Capítulo 5, Ejercicio 5.4: una regresión espuria con datos reales (oro y S&P 500)
import numpy as np, pandas as pd
import statsmodels.api as sm
from cap05_funciones import adf

s = pd.read_csv("datos/internacional/sp500_mensual_shiller.csv"); s["mes"] = s.Date.str[:7]
g = pd.read_csv("datos/internacional/oro_mensual.csv").rename(columns={"Date": "mes", "Price": "oro"})
d = s[["mes", "SP500"]].merge(g, on="mes")
d = d[(d.mes >= "1975-01") & (d.mes <= "2026-09")].reset_index(drop=True)
lo, ls = np.log(d.oro.to_numpy()), np.log(d.SP500.to_numpy())
print(f"T = {len(d)} meses ({d.mes.iloc[0]} a {d.mes.iloc[-1]})")
for n, x in [("lo", lo), ("ls", ls)]:
    print(f"ADF {n}: niveles t = {adf(x)['t']:.3f} | diferencias t = {adf(np.diff(x))['t']:.3f}")
m = sm.OLS(lo, sm.add_constant(ls)).fit(); e = m.resid
dw = np.sum(np.diff(e)**2) / np.sum(e**2)
print(f"\nNiveles:     b = {m.params[1]:.4f}  t = {m.tvalues[1]:.2f}  R2 = {m.rsquared:.4f}  DW = {dw:.4f}")
print(f"ADF de los residuos (sin constante): t = {adf(e, 'n')['t']:.3f}")
md = sm.OLS(np.diff(lo), sm.add_constant(np.diff(ls))).fit()
print(f"Diferencias: b = {md.params[1]:.4f}  t = {md.tvalues[1]:.2f}  R2 = {md.rsquared:.5f}")

# ---- Figura ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fecha = pd.to_datetime(d.mes)
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))
ax[0].plot(fecha, lo - lo[0], color=eg.OLIVA, label="log(oro), desde enero de 1975")
ax[0].plot(fecha, ls - ls[0], color=eg.AZUL, ls="--", label="log(S&P 500), desde enero de 1975")
ax[0].set_title("Niveles (ambos con raíz unitaria)"); ax[0].legend(fontsize=7.5)
ax[1].scatter(100 * np.diff(ls), 100 * np.diff(lo), s=5, color=eg.AZUL, alpha=0.5)
ax[1].set_xlabel("rendimiento mensual del S&P 500 (%)"); ax[1].set_ylabel("rendimiento mensual del oro (%)")
ax[1].set_title(f"Diferencias: $R^2={md.rsquared:.4f}$".replace(".", ","))
for a_ in ax: a_.yaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: f"{v_:g}".replace(".", ",")))
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: f"{v_:g}".replace(".", ",")))
fig.savefig("graficos/cap05_fig_oro.pdf")
