# Capítulo 14, Ejercicio 14.1: CAPM frente a Fama-French en las 9 carteras de tamaño y valor
import numpy as np
from cap14_funciones import datos, regs_ts, VALOR

d, Rall, F = datos(); R = Rall[VALOR].to_numpy(); T = len(R)
print(f"Muestra: {d[0]} a {d[-1]}, T = {T} meses; excesos de rendimiento en % mensual\n")
c1 = regs_ts(R, F[["MktRF"]]); c3 = regs_ts(R, F[["MktRF", "SMB", "HML"]])
print("Cartera   media   | CAPM: alfa  t(alfa)  beta   R2  | FF3: alfa  t(alfa)  b_Mkt   b_SMB   b_HML   R2")
for i, nm in enumerate(VALOR):
    print(f"{nm:<7} {R[:, i].mean():7.3f}   | {c1['alfa'][i]:8.3f} {c1['t_alfa'][i]:7.2f} {c1['beta'][i, 0]:6.2f} {c1['r2'][i]:5.2f} | "
          f"{c3['alfa'][i]:8.3f} {c3['t_alfa'][i]:7.2f} {c3['beta'][i, 0]:6.2f} {c3['beta'][i, 1]:7.2f} {c3['beta'][i, 2]:7.2f} {c3['r2'][i]:5.2f}")
print(f"\npromedio de |alfa|: CAPM = {np.abs(c1['alfa']).mean():.3f}; FF3 = {np.abs(c3['alfa']).mean():.3f} (% mensual)")
print(f"diferencia de medias S1V5 - S1V1 = {R[:, 2].mean() - R[:, 0].mean():.3f}; de alfas CAPM = {c1['alfa'][2] - c1['alfa'][0]:.3f}; "
      f"de alfas FF3 = {c3['alfa'][2] - c3['alfa'][0]:.3f}")

# ---- Figura: media realizada frente a la predicha por cada modelo ----
import estilo_graficos as eg, matplotlib.pyplot as plt
pred1 = c1["beta"] @ F[["MktRF"]].mean().to_numpy(); pred3 = c3["beta"] @ F[["MktRF", "SMB", "HML"]].mean().to_numpy()
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True, sharey=True)
for a, pr, tit in [(ax[0], pred1, "CAPM"), (ax[1], pred3, "Fama-French de tres factores")]:
    a.plot([0.3, 1.3], [0.3, 1.3], color=eg.GRIS, lw=0.8, ls="--"); a.plot(pr, R.mean(0), "o", color=eg.AZUL, ms=5)
    for i, nm in enumerate(VALOR): a.annotate(nm, (pr[i], R[:, i].mean()), (3, 3), textcoords="offset points", fontsize=7)
    a.set_title(tit, fontsize=9); a.set_xlabel("media predicha $\\hat\\beta_i'\\bar f$ (% mensual)"); eg.coma(a)
ax[0].set_ylabel("media realizada (% mensual)"); fig.tight_layout(); fig.savefig("graficos/cap14_ej01_predicho.pdf")
