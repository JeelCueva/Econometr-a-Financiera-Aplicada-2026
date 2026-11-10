# Capítulo 6, Ejercicio 6.2: estimación y diagnóstico de modelos ARMA para el logaritmo del VIX
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

v = pd.read_csv("datos/internacional/vix_diario.csv"); y = np.log(v["CLOSE"].to_numpy())
ordenes = {"AR(1)": (1, 0, 0), "AR(2)": (2, 0, 0), "ARMA(1,1)": (1, 0, 1)}
for n, o in ordenes.items():
    m = ARIMA(y, order=o, trend="c").fit(cov_type="approx"); k = o[0] + o[2]   # ee con el hessiano, como R
    lb = acorr_ljungbox(m.resid[1:], lags=[10, 20], model_df=k)   # sin el primer residuo
    nom = [x for x in m.model.param_names if x != "sigma2"]
    print(f"{n:<9} " + " ".join(f"{a}={b:.4f}" for a, b in zip(nom, m.params[:len(nom)])))
    print(f"          ee: " + " ".join(f"{b:.4f}" for b in m.bse[:len(nom)]))
    print(f"          AIC = {m.aic:.2f}  BIC = {m.bic:.2f}  Q(10) = {lb.lb_stat[10]:.1f} (p = {lb.lb_pvalue[10]:.3g})"
          f"  Q(20) = {lb.lb_stat[20]:.1f} (p = {lb.lb_pvalue[20]:.3g})")
m = ARIMA(y, order=(1, 0, 1), trend="c").fit(); phi = m.params[1]
print(f"\nARMA(1,1): sigma2 = {m.params[3]:.6f}  raíz AR = {1/phi:.4f}  vida media = {np.log(0.5)/np.log(phi):.1f} días")

# ---- Figura: ACF de los residuos de AR(1) y ARMA(1,1) ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
fig, ax = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True)
j = np.arange(1, 41); T = len(y)
for a, o, t in [(ax[0], (1, 0, 0), "Residuos del AR(1)"), (ax[1], (1, 0, 1), "Residuos del ARMA(1,1)")]:
    e = ARIMA(y, order=o, trend="c").fit().resid
    a.vlines(j, 0, acf(e, nlags=40)[1:], color=eg.AZUL, lw=1.5); a.axhline(0, color="black", lw=0.5)
    for sg in (1, -1): a.axhline(sg * 1.96 / np.sqrt(T), color=eg.LADRILLO, ls="--", lw=0.8)
    a.set_title(t); a.set_xlabel("rezago (días)"); a.set_ylim(-0.12, 0.12); eg.coma(a)
fig.savefig("graficos/cap06_fig_vix_resid.pdf")
