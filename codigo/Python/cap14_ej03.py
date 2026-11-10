# Capítulo 14, Ejercicio 14.3: primas por riesgo de Fama-MacBeth con 30 carteras
import numpy as np
from cap14_funciones import datos, fama_macbeth

d, R, F = datos(); T = len(R)
print(f"30 carteras (9 tamaño-valor, 9 tamaño-momentum, 12 industrias), T = {T} meses")
res = {}
for mn, fac in {"FF3": ["MktRF", "SMB", "HML"], "Carhart": ["MktRF", "SMB", "HML", "Mom"]}.items():
    fm = fama_macbeth(R, F[fac]); res[mn] = fm; nm = ["constante"] + fac
    print(f"\n{mn}: R2 de corte transversal = {fm['r2_ct']:.3f}; lambda' Omega^-1 lambda = {fm['c']:.4f}")
    print("              lambda    ee FM   t FM   ee Shanken  t Shanken   media del factor")
    for j, n in enumerate(nm):
        print(f"{n:<10} {fm['lam'][j]:9.4f} {fm['ee_fm'][j]:8.4f} {fm['lam'][j] / fm['ee_fm'][j]:6.2f} {fm['ee_sh'][j]:10.4f} "
              f"{fm['lam'][j] / fm['ee_sh'][j]:9.2f}   {'' if j == 0 else f'{F[fac[j - 1]].mean():.4f}'}")

# ---- Figura: media realizada frente a la ajustada por el corte transversal (Carhart) ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True, sharey=True)
grupos = [(slice(0, 9), eg.AZUL, "o", "tamaño-valor"), (slice(9, 18), eg.LADRILLO, "^", "tamaño-momentum"), (slice(18, 30), eg.OLIVA, "s", "industrias")]
for a, mn in zip(ax, ["FF3", "Carhart"]):
    fm = res[mn]; aj = np.column_stack([np.ones(30), fm["beta"]]) @ fm["lam"]; me = R.mean().to_numpy()
    a.plot([0.2, 1.4], [0.2, 1.4], color=eg.GRIS, lw=0.8, ls="--")
    for sl, col, mk, et in grupos: a.plot(aj[sl], me[sl], mk, color=col, ms=5, label=et)
    a.set_title(f"{mn}: $R^2$ = {fm['r2_ct']:.2f}".replace(".", ","), fontsize=9); a.set_xlabel("media ajustada (% mensual)"); eg.coma(a)
ax[0].set_ylabel("media realizada (% mensual)"); ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig("graficos/cap14_ej03_fama_macbeth.pdf")
