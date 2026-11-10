# Capítulo 15, Ejercicio 15.1: hechos estilizados de la curva de rendimientos y sus componentes principales
import numpy as np
from cap15_funciones import cargar, TAU
from statsmodels.tsa.stattools import acf

f, Y = cargar(); T = len(Y)
print(f"Rendimientos cupón cero de EE. UU. (McCulloch-Kwon), {f[0]} a {f[-1]}, T = {T} meses\n")
print("Vencimiento   media   desv.est.   autocorr(1)   autocorr(12)   corr. con 1 mes")
for j, m in enumerate(TAU):
    y = Y[:, j]; a = acf(y, nlags=12)
    print(f"{m:4d} meses   {y.mean():6.3f}   {y.std(ddof=1):7.3f}   {a[1]:10.4f}   {a[12]:11.4f}   {np.corrcoef(y, Y[:, 0])[0, 1]:12.4f}")
pend = Y[:, 9] - Y[:, 2]
print(f"\npendiente r120 - r3: media = {pend.mean():.3f}; meses con curva invertida (r120 < r3): {(pend < 0).sum()} ({100 * (pend < 0).mean():.1f} %)")
print(f"mínimo de la pendiente: {pend.min():.3f} ({f[pend.argmin()]}); máximo: {pend.max():.3f} ({f[pend.argmax()]})")
lam, V = np.linalg.eigh(np.cov(Y.T)); o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]; V = V[:, :3] * np.sign(V[9, :3])
print("\nComponentes principales (covarianza): % de la varianza total:", " ".join(f"{x:.2f}" for x in 100 * lam[:3] / lam.sum()))
print("Cargas (vencimiento: CP1  CP2  CP3)")
for j, m in enumerate(TAU): print(f"  {m:4d}: {V[j, 0]:7.3f} {V[j, 1]:7.3f} {V[j, 2]:7.3f}")

# ---- Figura: curvas en fechas seleccionadas y cargas de los componentes ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
for fe, col, ls in [("1955-06", eg.GRIS, ":"), ("1973-08", eg.OLIVA, "--"), ("1981-09", eg.LADRILLO, "-"), ("1986-12", eg.AZUL, "-.")]:
    i = int(np.flatnonzero(f == fe)[0]); ax[0].plot(TAU, Y[i], marker="o", ms=3, color=col, ls=ls, lw=1.2, label=fe)
ax[0].plot(TAU, Y.mean(0), color="black", lw=1.8, label="promedio 1946-1991")
ax[0].set_xlabel("vencimiento (meses)"); ax[0].set_ylabel("rendimiento (% anual)"); ax[0].legend(fontsize=7.5)
for k, (et, col, ls) in enumerate([("CP1: nivel", eg.AZUL, "-"), ("CP2: pendiente", eg.LADRILLO, "--"), ("CP3: curvatura", eg.OLIVA, ":")]):
    ax[1].plot(TAU, V[:, k], marker="o", ms=3, color=col, ls=ls, lw=1.3, label=et)
ax[1].axhline(0, color="black", lw=0.5); ax[1].set_xlabel("vencimiento (meses)"); ax[1].set_ylabel("carga"); ax[1].legend(fontsize=8)
for a in ax: eg.coma(a)
fig.tight_layout(); fig.savefig("graficos/cap15_ej01_curvas_cp.pdf")
