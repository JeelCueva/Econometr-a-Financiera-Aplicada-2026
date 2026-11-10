# Capítulo 11, Ejercicio 11.1: ¿están cointegrados el Brent y el WTI? Prueba de Engle-Granger
import numpy as np, pandas as pd, warnings
from cap11_funciones import crudo_mensual, engle_granger
from cap05_funciones import adf
from statsmodels.tsa.stattools import coint                # solo para comprobar
warnings.simplefilter("ignore")

m = crudo_mensual(); B = np.log(m["Price_b"].to_numpy()); W = np.log(m["Price_w"].to_numpy()); T = len(B); mes = m.index.to_numpy()
print(f"Promedios mensuales, {mes[0]} a {mes[-1]}, T = {T} meses\n")
print("ADF con constante (rezagos por AIC); valor crítico al 5 %: -2.86")
for et, y in [("log Brent", B), ("log WTI", W)]:
    a = adf(y, "c"); d = adf(np.diff(y), "c")
    print(f"  {et:<10} nivel: t = {a['t']:6.2f} (p = {a['p']})   diferencia: t = {d['t']:7.2f} (p = {d['p']})")
print("\nEngle-Granger:")
for dir_, (y, x) in [("Brent sobre WTI", (B, W)), ("WTI sobre Brent", (W, B))]:
    e = engle_granger(y, x)
    print(f"  {dir_}: a = {e['a']:.4f}, b = {e['b']:.4f}; ADF de residuos t = {e['t']:.3f} (p = {e['p']}), gamma = {e['gamma']:.4f}")
print("  valores críticos (MacKinnon, 2010):", " ".join(f"{k} {v:.3f}" for k, v in e["vc"].items()))
e = engle_granger(B, W); u = e["u"]
print(f"\nresiduo: desv. est. = {u.std(ddof=1):.4f}; mínimo {u.min():.4f} ({mes[u.argmin()]}); máximo {u.max():.4f} ({mes[u.argmax()]})")
print(f"vida media aproximada del desequilibrio: ln 0.5 / ln(1 + gamma) = {np.log(0.5) / np.log(1 + e['gamma']):.1f} meses")
c = coint(B, W); print(f"\ncomprobación con statsmodels (coint): t = {c[0]:.3f}, valor p = {c[1]:.6f}")

# ---- Figura: precios en logaritmos y residuo de la regresión de cointegración ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fe = pd.to_datetime(mes)
fig, ax = plt.subplots(2, 1, figsize=(9.0, 5.0), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
ax[0].plot(fe, B, color=eg.AZUL, lw=1, label="log Brent"); ax[0].plot(fe, W, color=eg.LADRILLO, lw=1, ls="--", label="log WTI")
ax[0].set_ylabel("log USD por barril"); ax[0].legend(fontsize=8)
ax[1].plot(fe, u, color=eg.GRIS, lw=0.9); ax[1].axhline(0, color="black", lw=0.5)
ax[1].set_ylabel("residuo $\\hat u_t$")
for a in ax: eg.coma(a)
ax[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
fig.savefig("graficos/cap11_ej01_brent_wti.pdf")
