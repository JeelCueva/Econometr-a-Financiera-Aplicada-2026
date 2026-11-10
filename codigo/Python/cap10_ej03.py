# Capítulo 10, Ejercicio 10.3: respuestas al impulso del S&P 500 y del VIX; sensibilidad al orden de Cholesky
import numpy as np, pandas as pd
from cap10_funciones import var_mco, irf, bootstrap_irf
from statsmodels.tsa.api import VAR                       # solo para comprobar

s = pd.read_csv("datos/internacional/sp500_diario_yahoo.csv"); v = pd.read_csv("datos/internacional/vix_diario.csv")
d = s[["Date", "Adj Close"]].merge(v[["DATE", "CLOSE"]], left_on="Date", right_on="DATE")
Y = np.column_stack([100 * np.diff(np.log(d["Adj Close"])), 100 * np.diff(np.log(d["CLOSE"]))])
m = var_mco(Y, 2); P = np.linalg.cholesky(m["S"])        # triangular inferior
print("Factor de Cholesky P (orden r, dv):"); print(np.round(P, 4))
th = irf(m["A"], 10, P)
print("\nOrden (r, dv). Respuestas ortogonalizadas:\n  h    r<-r    dv<-r    r<-dv   dv<-dv")
for h in (0, 1, 2, 5, 10): T = th[h]; print(f"{h:3d} {T[0, 0]:7.4f} {T[1, 0]:8.4f} {T[0, 1]:8.4f} {T[1, 1]:8.4f}")
mr = var_mco(Y[:, ::-1], 2); thr = irf(mr["A"], 10, np.linalg.cholesky(mr["S"]))
print("\nOrden (dv, r). Respuestas (variables en el orden r, dv):\n  h    r<-r    dv<-r    r<-dv   dv<-dv")
for h in (0, 1, 2, 5, 10):
    T = thr[h][::-1, ::-1]; print(f"{h:3d} {T[0, 0]:7.4f} {T[1, 0]:8.4f} {T[0, 1]:8.4f} {T[1, 1]:8.4f}")
print(f"\nrespuesta acumulada (10 días) del VIX a un choque de +1 desv. est. en r: {th.sum(axis=0)[1, 0]:.3f} %")
lo, hi = bootstrap_irf(Y, 2, 10, B=500, semilla=2026)
print("Bandas bootstrap al 90 % (500 réplicas), orden (r, dv):")
for h in (0, 1, 2): print(f"  h = {h}: dv<-r [{lo[h, 1, 0]:.3f}, {hi[h, 1, 0]:.3f}]; r<-dv [{lo[h, 0, 1]:.4f}, {hi[h, 0, 1]:.4f}]")
print(f"\ncomprobación con statsmodels: respuestas iguales = {np.allclose(VAR(Y).fit(2).orth_ma_rep(10), th)}")

# ---- Figura: respuestas con bandas (orden r, dv) y respuesta puntual con el orden inverso ----
import estilo_graficos as eg, matplotlib.pyplot as plt
thr2 = thr[:, ::-1, ::-1]; hh = np.arange(11); et = ["r", "\\Delta v"]
fig, ax = plt.subplots(2, 2, figsize=(9.0, 5.4), sharex=True)
for k in range(2):
    for j in range(2):
        a = ax[k, j]
        a.fill_between(hh, lo[:, k, j], hi[:, k, j], color=eg.AZUL, alpha=0.18, label="banda bootstrap 90 %")
        a.plot(hh, th[:, k, j], color=eg.AZUL, lw=1.5, marker="o", ms=3, label="orden $(r, \\Delta v)$")
        a.plot(hh, thr2[:, k, j], color=eg.LADRILLO, lw=1.2, ls="--", label="orden $(\\Delta v, r)$")
        a.axhline(0, color="black", lw=0.5); a.set_title(f"respuesta de ${et[k]}$ al choque en ${et[j]}$"); eg.coma(a)
ax[1, 0].set_xlabel("días"); ax[1, 1].set_xlabel("días"); ax[0, 1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("graficos/cap10_ej03_irf.pdf")
