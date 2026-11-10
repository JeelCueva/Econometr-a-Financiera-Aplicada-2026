# Capítulo 7, Ejercicio 7.3: reversión a la media de largo plazo (S&P, rendimiento real anual, Shiller)
import numpy as np, pandas as pd
from cap07_funciones import vr_lm

s = pd.read_csv("datos/internacional/sp500_mensual_shiller.csv", parse_dates=["Date"])
s = s[s["Date"] <= "2023-06-01"].reset_index(drop=True)
P, D, I = s["SP500"].to_numpy(), s["Dividend"].to_numpy(), s["Consumer Price Index"].to_numpy()
g = np.log((P[1:] + D[1:] / 12) / P[:-1]) - np.diff(np.log(I))   # rendimiento real total mensual
idx = np.concatenate([[0.0], np.cumsum(g)])                       # índice de riqueza real (log)
ene = (s["Date"].dt.month == 1).to_numpy()
ra = np.diff(idx[ene]); T = len(ra); anios = s["Date"][ene].dt.year.to_numpy()
print(f"T = {T} rendimientos reales anuales (enero a enero, {anios[0]} a {anios[-1]})")
rho1 = ((ra[1:] - ra.mean()) * (ra[:-1] - ra.mean())).sum() / ((ra - ra.mean())**2).sum()
print(f"media = {ra.mean():.4f}, desv. est. = {ra.std(ddof=1):.4f}, rho1 = {rho1:.3f}")
print("   q      VR        z       z*")
for q in [2, 3, 5, 8, 10]:
    v = vr_lm(ra, q); print(f"{q:4d} {v['VR']:7.3f} {v['z']:8.2f} {v['zrob']:8.2f}")

# ---- Figura: VR(q) de largo plazo con banda robusta y desviaciones del índice real ----
import estilo_graficos as eg, matplotlib.pyplot as plt
qs = np.arange(2, 16); res = [vr_lm(ra, q) for q in qs]
vr = np.array([x["VR"] for x in res]); ee = (vr - 1) / np.array([x["zrob"] for x in res])
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
la = idx[ene]; t = np.arange(len(la)); b = np.polyfit(t, la, 1)
ax[0].plot(anios, la - np.polyval(b, t), color=eg.AZUL, lw=1)
ax[0].axhline(0, color="black", lw=0.6)
ax[0].set_title("Log riqueza real menos tendencia lineal"); ax[0].set_xlabel("año")
ax[1].fill_between(qs, 1 - 1.96 * ee, 1 + 1.96 * ee, color=eg.AZUL, alpha=0.15, label="banda 95 % robusta")
ax[1].plot(qs, vr, color=eg.LADRILLO, marker="o", ms=3, label="$\\widehat{VR}(q)$")
ax[1].axhline(1, color="black", lw=0.6); ax[1].set_xlabel("horizonte $q$ (años)")
ax[1].set_title("Razón de varianzas anual"); ax[1].legend(fontsize=8)
for a in ax: eg.coma(a)
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
ax[1].set_xticks(range(2, 16, 2))
fig.savefig("graficos/cap07_ej03_reversion.pdf")
