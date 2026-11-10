# Capítulo 10, Ejercicio 10.4: VAR mensual de inflación, tasa de interés y rendimiento bursátil (EE. UU., 1960-2023)
import numpy as np, pandas as pd
from cap10_funciones import seleccion, granger, var_mco, irf, fevd, bootstrap_irf
from statsmodels.tsa.api import VAR                       # solo para comprobar

s = pd.read_csv("datos/internacional/sp500_mensual_shiller.csv")
s = s[(s["Date"] >= "1960-01-01") & (s["Date"] <= "2023-06-01")].reset_index(drop=True)
P, D, I, R = (s[c].to_numpy() for c in ["SP500", "Dividend", "Consumer Price Index", "Long Interest Rate"])
Y = np.column_stack([100 * np.diff(np.log(I)), np.diff(R), 100 * np.log((P[1:] + D[1:] / 12) / P[:-1])])
print(f"T = {len(Y)} meses ({s['Date'].iloc[1][:7]} a {s['Date'].iloc[-1][:7]})")
print("medias:", " ".join(f"{x:.4f}" for x in Y.mean(0)), "; desv. est.:", " ".join(f"{x:.4f}" for x in Y.std(0, ddof=1)))
ic = seleccion(Y, 12)
print(f"orden elegido: AIC = {int(ic[ic[:, 1].argmin(), 0])}, BIC = {int(ic[ic[:, 2].argmin(), 0])}, HQ = {int(ic[ic[:, 3].argmin(), 0])}\n")
p = 2; nom = ["inf", "dtasa", "ret"]
print("Causalidad de Granger, VAR(2) (F de una ecuación y Wald HC0):")
for e in range(3):
    for c in [x for x in range(3) if x != e]:
        g = granger(Y, p, e, [c])
        print(f"  {nom[c]:<5} no causa a {nom[e]:<5} F = {g['F']:6.3f} (p = {g['pF']:.4f}); Wald HC0 = {g['W']:6.3f} (p = {g['pW']:.4f})")
g = granger(Y, p, 2, [0, 1])
print(f"  inf y dtasa juntas no causan a ret: F = {g['F']:.3f} (p = {g['pF']:.4f}); Wald HC0 = {g['W']:.3f} (p = {g['pW']:.4f})")
m = var_mco(Y, p); th = irf(m["A"], 24, np.linalg.cholesky(m["S"]))
print("\nRespuesta de ret (%) a choques de una desv. est. (orden inf, dtasa, ret):\n  h      inf    dtasa      ret")
for h in (0, 1, 2, 3, 6, 12): print(f"{h:3d} {th[h, 2, 0]:8.4f} {th[h, 2, 1]:8.4f} {th[h, 2, 2]:8.4f}")
ac = th.sum(axis=0); print(f"respuesta acumulada de ret a 24 meses: a inf = {ac[2, 0]:.3f}; a dtasa = {ac[2, 1]:.3f}")
fv = fevd(th)
print("\nDescomposición de la varianza del error de pronóstico de ret (%):\n  h      inf    dtasa      ret")
for h in (1, 3, 6, 12, 24): print(f"{h:3d} {100 * fv[h - 1, 2, 0]:8.2f} {100 * fv[h - 1, 2, 1]:8.2f} {100 * fv[h - 1, 2, 2]:8.2f}")
print("\nDescomposición para inf a 24 meses (%):", " ".join(f"{100 * x:.2f}" for x in fv[23, 0]))
r = VAR(Y).fit(2)
print(f"\ncomprobación con statsmodels: FEVD igual = {np.allclose(r.fevd(24).decomp, fv[:24].transpose(1, 0, 2))}")

# ---- Figura: respuestas de ret con bandas y descomposición de su varianza ----
import estilo_graficos as eg, matplotlib.pyplot as plt
lo, hi = bootstrap_irf(Y, p, 24, B=500, semilla=2026); hh = np.arange(25)
fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.3))
for a, j, tit in [(ax[0], 0, "choque de inflación"), (ax[1], 1, "choque de tasa de interés")]:
    a.fill_between(hh, lo[:, 2, j], hi[:, 2, j], color=eg.AZUL, alpha=0.18)
    a.plot(hh, th[:, 2, j], color=eg.AZUL, lw=1.4, marker="o", ms=2.5); a.axhline(0, color="black", lw=0.5)
    a.set_title(f"ret ante un {tit}", fontsize=9); a.set_xlabel("meses"); eg.coma(a)
ax[0].set_ylabel("puntos porcentuales")
h = np.arange(1, 25); base = np.zeros(24)
for j, et, col in [(2, "propio ret", eg.GRIS), (0, "inflación", eg.LADRILLO), (1, "tasa", eg.OLIVA)]:
    ax[2].bar(h, 100 * fv[:24, 2, j], bottom=base, color=col, width=0.8, label=et); base += 100 * fv[:24, 2, j]
ax[2].set_ylim(90, 100); ax[2].set_title("varianza del error de ret (%)", fontsize=9); ax[2].set_xlabel("horizonte (meses)")
ax[2].legend(fontsize=7, loc="lower right", framealpha=0.9); eg.coma(ax[2])
fig.tight_layout(); fig.savefig("graficos/cap10_ej04_macro.pdf")
