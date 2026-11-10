# Capítulo 4, Ejercicio 4.4: ¿predice la razón dividendo/precio el rendimiento a 12 meses?
import numpy as np, pandas as pd
import statsmodels.api as sm

s = pd.read_csv("datos/internacional/sp500_mensual_shiller.csv", parse_dates=["Date"])
s = s[s.Date <= "2023-06-01"].reset_index(drop=True)      # último mes con dividendos
P, D = s.SP500.to_numpy(), s.Dividend.to_numpy(); n, k = len(P), 12
g = np.log((P[1:] + D[1:] / 12) / P[:-1])                 # rendimiento total mensual (log)
dp = np.log(D / P)
y = np.array([g[t:t + k].sum() for t in range(n - k)])   # rendimiento de t a t+12
x = dp[:n - k]; T = len(y); X = sm.add_constant(x)
cl = sm.OLS(y, X).fit()
print(f"Muestra: {s.Date[0].date()} a {s.Date[T-1].date()} (T = {T} regresiones traslapadas)")
print(f"b = {cl.params[1]:.4f}   R2 = {cl.rsquared:.4f}")
tab = {"clasico": cl.bse[1]}
for L in (12, 24):
    tab[f"NW_{L}"] = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": False}).bse[1]
tab = pd.DataFrame({"ee_b": tab}); tab["t_b"] = cl.params[1] / tab.ee_b
print(tab.round(3).to_string())
enero = (s.Date[:T].dt.month == 1).to_numpy()
m = sm.OLS(y[enero], X[enero]).fit()
print(f"Sin traslape (eneros, T = {enero.sum()}): b = {m.params[1]:.4f}  ee = {m.bse[1]:.4f}  t = {m.tvalues[1]:.2f}")
print(f"Autocorrelación de dp: rezago 1 = {np.corrcoef(dp[1:], dp[:-1])[0,1]:.3f}, "
      f"rezago 12 = {np.corrcoef(dp[12:], dp[:-12])[0,1]:.3f}")

# ---- Figura: predictor y rendimiento futuro ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))
fechas = s.Date[:T]
ax[0].plot(fechas, 100 * np.exp(x), color=eg.AZUL, lw=0.9)
ax[0].set_ylabel("dividendo / precio (%)"); ax[0].set_title("Rendimiento por dividendo del S&P 500")
ax[1].scatter(x, 100 * y, s=4, color=eg.AZUL, alpha=0.35)
gg = np.linspace(x.min(), x.max(), 2); ax[1].plot(gg, 100 * (cl.params[0] + cl.params[1] * gg), color=eg.LADRILLO)
ax[1].set_xlabel(r"$\ln(D_t/P_t)$"); ax[1].set_ylabel("rendimiento de $t$ a $t+12$ (%)")
ax[1].set_title("Regresión predictiva (meses traslapados)")
ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))); eg.coma(ax[1])
fig.savefig("graficos/cap04_fig_dividendos.pdf")
