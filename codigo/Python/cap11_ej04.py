# Capítulo 11, Ejercicio 11.4: el modelo de valor presente: ¿están cointegrados precios, dividendos y utilidades?
import numpy as np, pandas as pd, warnings
from scipy import stats
from cap11_funciones import johansen, vecm, lr_beta
from cap05_funciones import adf
from cap10_funciones import seleccion
warnings.simplefilter("ignore")

s = pd.read_csv("datos/internacional/sp500_mensual_shiller.csv")
s = s[(s["Date"] <= "2023-06-01") & (s["Date"].str[5:7] == "01")].reset_index(drop=True)   # enero de cada año
P, D, E = (np.log(s[c].to_numpy()) for c in ["Real Price", "Real Dividend", "Real Earnings"]); T = len(P)
anio = s["Date"].str[:4].to_numpy()
print(f"Datos anuales (enero), {anio[0]} a {anio[-1]}, T = {T}; valores reales\n")
print("ADF con constante (rezagos por AIC); valor crítico al 5 %: -2.86")
for et, y in [("log P", P), ("log D", D), ("log E", E), ("log(D/P)", D - P), ("log(E/P)", E - P)]:
    a = adf(y, "c"); print(f"  {et:<9} t = {a['t']:6.2f} (p = {a['p']})")
for par, X in [("D", D), ("E", E)]:
    Y = np.column_stack([P, X]); ic = seleccion(Y, 4); k = int(ic[ic[:, 2].argmin(), 0]) - 1
    J = johansen(Y, k); v = vecm(Y, k); LR, *_ = lr_beta(Y, k, np.array([[1.0], [-1.0]]))
    a, ea = v["alpha"][:, 0], v["ee_alpha"][:, 0]
    print(f"\nPrecio y {par}: k = {k}; traza r=0: {J['traza'][0]:.2f} (VC 95 % 15.49); r<=1: {J['traza'][1]:.2f} (VC 3.84)")
    print(f"  beta = (1, {v['beta'][1, 0]:.4f}); alpha_P = {a[0]:.4f} (t = {a[0] / ea[0]:.2f}); alpha_{par} = {a[1]:.4f} (t = {a[1] / ea[1]:.2f})")
    print(f"  LR beta = (1, -1): {LR:.2f} (p = {stats.chi2.sf(LR, 1):.4f})")
print(f"\nrazón dividendo/precio: {100 * np.exp(D[0] - P[0]):.2f} % en {anio[0]} y {100 * np.exp(D[-1] - P[-1]):.2f} % en {anio[-1]}")
print(f"razón de pago D/E promedio: 1871-1950 = {100 * np.exp(D - E)[anio <= '1950'].mean():.1f} %; "
      f"1990-2023 = {100 * np.exp(D - E)[anio >= '1990'].mean():.1f} %")

# ---- Figura: razones dividendo/precio y utilidad/precio ----
import estilo_graficos as eg, matplotlib.pyplot as plt
an = anio.astype(int)
fig, ax = plt.subplots(figsize=(9.0, 3.5))
ax.plot(an, 100 * np.exp(D - P), color=eg.AZUL, lw=1.2, label="dividendo / precio")
ax.plot(an, 100 * np.exp(E - P), color=eg.LADRILLO, lw=1.2, ls="--", label="utilidad / precio")
ax.set_ylabel("%"); ax.legend(fontsize=8); eg.coma(ax)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
fig.savefig("graficos/cap11_ej04_valor_presente.pdf")
