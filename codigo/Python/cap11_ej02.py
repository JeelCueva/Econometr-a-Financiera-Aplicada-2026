# Capítulo 11, Ejercicio 11.2: prueba de Johansen y VECM para el Brent y el WTI
import numpy as np, warnings
from scipy import stats
from cap11_funciones import crudo_mensual, johansen, vecm, lr_beta, VC_TRAZA, VC_MAXV
from cap10_funciones import seleccion
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM   # solo para comprobar
warnings.simplefilter("ignore")

m = crudo_mensual(); L = np.log(m[["Price_b", "Price_w"]].to_numpy()); nom = ["Brent", "WTI"]
ic = seleccion(L, 8); p = int(ic[ic[:, 2].argmin(), 0]); k = p - 1
print(f"VAR en niveles: BIC elige p = {p} (AIC elige {int(ic[ic[:, 1].argmin(), 0])}), de modo que k = p - 1 = {k} rezagos de las diferencias\n")
J = johansen(L, k); K = 2
print(f"autovalores: {J['lam'][0]:.5f}  {J['lam'][1]:.5f}  (n = {J['n']})")
print("H0          traza   VC 95 %    máx. autovalor   VC 95 %")
for r in range(2):
    print(f"r <= {r}   {J['traza'][r]:8.3f}  {VC_TRAZA[K - r][1]:8.3f}        {J['maxv'][r]:8.3f}      {VC_MAXV[K - r][1]:8.3f}")
v = vecm(L, k, 1)
print(f"\nvector de cointegración normalizado: beta = (1, {v['beta'][1, 0]:.4f})")
print("coeficientes de ajuste:")
for i in range(2):
    a, e = v["alpha"][i, 0], v["ee_alpha"][i, 0]; print(f"  alpha_{nom[i]:<5} = {a:8.4f} (ee {e:.4f}, t = {a / e:5.2f})")
rho = 1 + (v["beta"][:, 0] * v["alpha"][:, 0]).sum()
print(f"persistencia del desequilibrio 1 + beta'alpha = {rho:.4f}; vida media = {np.log(0.5) / np.log(rho):.1f} meses")
LR, *_ = lr_beta(L, k, np.array([[1.0], [-1.0]]))
print(f"\nH0: beta = (1, -1): LR = {LR:.3f}, p = {stats.chi2.sf(LR, 1):.2e} (chi-cuadrado con 1 g.l.)")
c = coint_johansen(L, 0, k); s = VECM(L, k_ar_diff=k, coint_rank=1, deterministic="co").fit()
print(f"\ncomprobación con statsmodels: traza = {np.round(c.lr1, 3)}, máx. = {np.round(c.lr2, 3)}; "
      f"beta = {np.round(s.beta.ravel(), 4)}; alpha = {np.round(s.alpha.ravel(), 4)}")

# ---- Figura: término de corrección de errores y ajuste de cada precio ----
import pandas as pd, estilo_graficos as eg, matplotlib.pyplot as plt
fe = pd.to_datetime(m.index.to_numpy()[k + 1:]); ect = v["ect"][:, 0] - v["ect"][:, 0].mean()
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
ax[0].plot(fe, ect, color=eg.GRIS, lw=0.9); ax[0].axhline(0, color="black", lw=0.5)
ax[0].set_title("Desequilibrio $\\hat\\beta' y_{t-1}$ (centrado)", fontsize=9)
for i, col in enumerate([eg.AZUL, eg.LADRILLO]):
    ax[1].plot(ect, v["J"]["dY"][:, i], ".", color=col, ms=2.5, alpha=0.6, label=f"$\\Delta$ log {nom[i]}")
    xs = np.linspace(ect.min(), ect.max(), 10); ax[1].plot(xs, v["alpha"][i, 0] * xs, color=col, lw=1.5)
ax[1].set_xlabel("desequilibrio del mes anterior"); ax[1].set_title("Pendientes $\\hat\\alpha_i$", fontsize=9)
ax[1].legend(fontsize=8); ax[1].set_ylim(-0.4, 0.4)
for a in ax: eg.coma(a)
ax[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
fig.tight_layout(); fig.savefig("graficos/cap11_ej02_vecm.pdf")
