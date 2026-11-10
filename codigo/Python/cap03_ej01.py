# Capítulo 3, Ejercicio 3.1: beta del sector de servicios públicos (Utils), paso a paso
import numpy as np, pandas as pd
import statsmodels.api as sm

d = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv")
y = (d["Utils"] - d["RF"]).to_numpy()       # exceso de rendimiento del sector
x = d["MktRF"].to_numpy()                   # exceso de rendimiento del mercado
T = len(y)

Sxx = np.sum((x - x.mean())**2); Sxy = np.sum((x - x.mean()) * (y - y.mean()))
beta = Sxy / Sxx
alfa = y.mean() - beta * x.mean()
e = y - alfa - beta * x
s2 = np.sum(e**2) / (T - 2)
ee_b = np.sqrt(s2 / Sxx)
ee_a = np.sqrt(s2 * (1 / T + x.mean()**2 / Sxx))
R2 = 1 - np.sum(e**2) / np.sum((y - y.mean())**2)
print(f"T = {T} meses ({d.dates.iloc[0]} a {d.dates.iloc[-1]})")
print(f"beta = {beta:.4f}  ee = {ee_b:.4f}  t(beta=1) = {(beta-1)/ee_b:.2f}")
print(f"alfa = {alfa:.5f} mensual ({1200*alfa:.2f} % anual)  ee = {ee_a:.5f}  t(alfa=0) = {alfa/ee_a:.2f}")
print(f"R2 = {R2:.4f}")

v_tot, v_sis, v_idi = 12 * y.var(ddof=1), 12 * beta**2 * x.var(ddof=1), 12 * e.var(ddof=1)
print(f"Varianza anual: total {v_tot:.5f} = sistemática {v_sis:.5f} + idiosincrática {v_idi:.5f}")
print(f"Volatilidad anual: total {100*np.sqrt(v_tot):.2f} %, sistemática {100*np.sqrt(v_sis):.2f} %, "
      f"idiosincrática {100*np.sqrt(v_idi):.2f} %")

print("\nComprobación con statsmodels:")
mod = sm.OLS(y, sm.add_constant(x)).fit()
print(pd.DataFrame({"coef": mod.params, "ee": mod.bse, "t": mod.tvalues, "p": mod.pvalues},
                   index=["const", "x"]).round(5).to_string())

# ---- Figura: nube de puntos, recta MCO y recta de beta = 1 ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.scatter(100 * x, 100 * y, s=6, color=eg.AZUL, alpha=0.5, label="meses, 1949-2017")
g = np.linspace(x.min(), x.max(), 2)
ax.plot(100 * g, 100 * (alfa + beta * g), color=eg.LADRILLO, lw=1.6, label=f"MCO: $\\hat\\beta={beta:.3f}$".replace(".", ","))
ax.plot(100 * g, 100 * g, color=eg.GRIS, ls="--", lw=1.1, label="$\\beta=1$")
ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
ax.set_xlabel("exceso de rendimiento del mercado (%)"); ax.set_ylabel("exceso de rendimiento de Utils (%)")
ax.legend(fontsize=8); eg.coma(ax)
fig.savefig("graficos/cap03_fig_utils.pdf")
