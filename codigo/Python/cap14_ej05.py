# Capítulo 14, Ejercicio 14.5: factores estadísticos: componentes principales de 30 carteras
import numpy as np
from cap14_funciones import datos, regs_ts, VALOR, MOMENTUM, INDUSTRIAS

d, Rdf, F = datos(); R = Rdf.to_numpy(); T = len(R)
lam, V = np.linalg.eigh(np.cov(R.T, bias=True)); o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
V = V * np.sign(V.sum(0))                                   # signo: cargas que sumen positivo
print("Proporción de la varianza total explicada:")
for k in range(5): print(f"  CP{k + 1}: {100 * lam[k] / lam.sum():6.2f} %  (acumulada {100 * lam[:k + 1].sum() / lam.sum():6.2f} %)")
P = (R - R.mean(0)) @ V[:, :3]; Fn = F.to_numpy()
print("\nCorrelación de los tres primeros componentes con los factores:\n          Mkt     SMB     HML     Mom")
for k in range(3): print(f"CP{k + 1}   " + "  ".join(f"{np.corrcoef(P[:, k], Fn[:, j])[0, 1]:6.3f}" for j in range(4)))
r2 = [regs_ts(P[:, [k]], Fn)["r2"][0] for k in range(3)]
print("\nR2 de cada componente explicado por los cuatro factores:", " ".join(f"{x:.3f}" for x in r2))
print(f"\nR2 medio de las 30 carteras: con 3 componentes = {regs_ts(R, P)['r2'].mean():.3f}; con Mkt, SMB y HML = {regs_ts(R, Fn[:, :3])['r2'].mean():.3f}")

# ---- Figura: varianza explicada y cargas de los dos primeros componentes ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1, 2]})
ax[0].bar(np.arange(1, 11), 100 * lam[:10] / lam.sum(), color=eg.AZUL); ax[0].set_xlabel("componente"); ax[0].set_ylabel("% de la varianza total")
ax[0].set_xticks(range(1, 11)); eg.coma(ax[0])
nom = VALOR + MOMENTUM + INDUSTRIAS; x = np.arange(30)
ax[1].bar(x - 0.2, V[:, 0], 0.4, color=eg.AZUL, label="CP1"); ax[1].bar(x + 0.2, V[:, 1], 0.4, color=eg.LADRILLO, label="CP2")
ax[1].set_xticks(x); ax[1].set_xticklabels(nom, rotation=90, fontsize=6.5); ax[1].axhline(0, color="black", lw=0.5)
ax[1].legend(fontsize=8); ax[1].set_ylabel("carga"); eg.coma(ax[1]); ax[1].set_xticklabels(nom, rotation=90, fontsize=6.5)
fig.tight_layout(); fig.savefig("graficos/cap14_ej05_componentes.pdf")
