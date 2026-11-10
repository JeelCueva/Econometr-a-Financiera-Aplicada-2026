# Capítulo 6, Ejercicio 6.5: el sesgo de Stambaugh en la regresión predictiva con la razón dividendo/precio
import numpy as np, pandas as pd
import statsmodels.api as sm

s = pd.read_csv("datos/internacional/sp500_mensual_shiller.csv", parse_dates=["Date"])
s = s[s.Date <= "2023-06-01"].reset_index(drop=True)
P, D = s.SP500.to_numpy(), s.Dividend.to_numpy()
r = np.log((P[1:] + D[1:] / 12) / P[:-1]); dp = np.log(D / P)
usar = (s.Date[:-1] >= "1946-01-01").to_numpy()
x, x1, r = dp[:-1][usar], dp[1:][usar], r[usar]; T = len(r)
m1 = sm.OLS(r, sm.add_constant(x)).fit(); m2 = sm.OLS(x1, sm.add_constant(x)).fit()
b, phi = m1.params[1], m2.params[1]; u, v = m1.resid, m2.resid
k = np.cov(u, v)[0, 1] / np.var(v, ddof=1)
sesgo_phi = -(1 + 3 * phi) / T; sesgo_b = k * sesgo_phi
print(f"T = {T} meses ({s.Date[np.where(usar)[0][0]].date()} a {s.Date[np.where(usar)[0][-1]].date()})")
print(f"b = {b:.5f} (t = {m1.tvalues[1]:.2f})   phi = {phi:.5f}   corr(u, v) = {np.corrcoef(u, v)[0,1]:.3f}   "
      f"sigma_uv/sigma_v^2 = {k:.4f}")
print(f"Sesgo de Kendall de phi = {sesgo_phi:.5f}\nSesgo de Stambaugh de b = {sesgo_b:.5f}\nb corregido = {b - sesgo_b:.5f}")
print(f"Proporción del coeficiente atribuible al sesgo = {100 * sesgo_b / b:.1f} %")

# ---- Figura: innovaciones del rendimiento y del predictor ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.4, 3.8))
ax.scatter(100 * v, 100 * u, s=5, color=eg.AZUL, alpha=0.4)
g = np.linspace(v.min(), v.max(), 2); ax.plot(100 * g, 100 * k * g, color=eg.LADRILLO, lw=1.4)
ax.set_xlabel("innovación del log D/P, $v_t$ (×100)"); ax.set_ylabel("innovación del rendimiento, $u_t$ (%)")
ax.set_title(f"corr$(u,v)={np.corrcoef(u, v)[0,1]:.3f}$".replace(".", ",")); eg.coma(ax)
fig.savefig("graficos/cap06_fig_stambaugh.pdf")
