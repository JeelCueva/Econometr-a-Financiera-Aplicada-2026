# Capítulo 7, Ejercicio 7.1: razón de varianzas del Brent diario
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import acf
from cap07_funciones import vr_lm
from arch.unitroot import VarianceRatio                   # solo para comprobar (recibe el log precio)

b = pd.read_csv("datos/internacional/brent_diario.csv")
lp = np.log(b["Price"].to_numpy()); r = np.diff(lp); T = len(r)
print(f"T = {T} rendimientos diarios ({b['Date'].iloc[1]} a {b['Date'].iloc[-1]})")
print("ACF 1-5:", " ".join(f"{v:6.3f}" for v in acf(r, nlags=5)[1:]))
print("   q       VR        z       z*   | arch: z     z*")
for q in [2, 5, 10, 20]:
    v = vr_lm(r, q)
    a0 = VarianceRatio(lp, lags=q, robust=False).stat; a1 = VarianceRatio(lp, lags=q).stat
    print(f"{q:4d} {v['VR']:8.4f} {v['z']:8.2f} {v['zrob']:8.2f}   | {a0:8.2f} {a1:6.2f}")

# ---- Figura: VR(q) para q = 2,...,40 con banda robusta del 95 % ----
import estilo_graficos as eg, matplotlib.pyplot as plt
qs = np.arange(2, 41); res = [vr_lm(r, q) for q in qs]
vr = np.array([x["VR"] for x in res]); ee = (vr - 1) / np.array([x["zrob"] for x in res])
ee0 = np.sqrt(2 * (2 * qs - 1) * (qs - 1) / (3 * qs * T))
fig, ax = plt.subplots(figsize=(6.2, 3.3))
ax.fill_between(qs, 1 - 1.96 * ee, 1 + 1.96 * ee, color=eg.AZUL, alpha=0.15, label="banda 95 % robusta ($z^*$)")
ax.plot(qs, 1 - 1.96 * ee0, color=eg.GRIS, ls=":", lw=0.9, label="banda 95 % homocedástica ($z$)")
ax.plot(qs, 1 + 1.96 * ee0, color=eg.GRIS, ls=":", lw=0.9)
ax.plot(qs, vr, color=eg.LADRILLO, marker="o", ms=3, label="$\\widehat{VR}(q)$ Brent")
ax.axhline(1, color="black", lw=0.6); ax.set_xlabel("horizonte $q$ (días)"); ax.set_ylabel("$VR(q)$")
ax.legend(fontsize=8); eg.coma(ax); fig.savefig("graficos/cap07_ej01_vr_brent.pdf")
