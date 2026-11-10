# Capítulo 5, Ejercicio 5.1: ¿tiene raíz unitaria el precio del Brent? ¿Y sus rendimientos?
import numpy as np, pandas as pd
from cap05_funciones import adf, kpss, VC_ADF, VC_KPSS
from arch.unitroot import ADF, KPSS                       # solo para comprobar

b = pd.read_csv("datos/internacional/brent_diario.csv", parse_dates=["Date"])
lp = np.log(b["Price"].to_numpy()); r = np.diff(lp)
casos = {"log precio (c)": (lp, "c"), "log precio (ct)": (lp, "ct"), "rendimiento (c)": (r, "c")}
for n, (y, det) in casos.items():
    a, k = adf(y, det), kpss(y, det)
    print(f"{n:<16} ADF: t = {a['t']:8.3f} (p = {a['p']:2d} rezagos; VC 5% = {VC_ADF[det]['5%']:5.2f}) | "
          f"KPSS: eta = {k['eta']:6.3f} (l = {k['l']}; VC 5% = {VC_KPSS[det]['5%']:.3f})")
    chk_a = ADF(y, lags=a["p"], trend=det); chk_k = KPSS(y, lags=k["l"], trend=det)
    print(f"{'':16} comprobación con arch: ADF {chk_a.stat:8.3f} (p-valor {chk_a.pvalue:.3f}) | "
          f"KPSS {chk_k.stat:6.3f} (p-valor {chk_k.pvalue:.3f})")

# ---- Figura ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
fig, ax = plt.subplots(2, 2, figsize=(9.6, 5.2), gridspec_kw={"width_ratios": [2.3, 1]})
f = b["Date"]
ax[0, 0].plot(f, lp, color=eg.LADRILLO, lw=0.7); ax[0, 0].set_title("Logaritmo del precio del Brent")
ax[1, 0].plot(f[1:], r, color=eg.AZUL, lw=0.4); ax[1, 0].set_title("Rendimiento logarítmico diario")
ax[1, 0].set_ylim(-0.3, 0.3)
for i, y in enumerate([lp, r]):
    ax[i, 1].vlines(range(1, 101), 0, acf(y, nlags=100)[1:], color=eg.LADRILLO if i == 0 else eg.AZUL)
    ax[i, 1].axhline(0, color="black", lw=0.5); ax[i, 1].set_ylim(-0.1, 1.02); ax[i, 1].set_title("ACF")
ax[1, 1].set_xlabel("rezago (días)")
for a_ in ax.flat: a_.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.tight_layout(); fig.savefig("graficos/cap05_fig_brent.pdf")
