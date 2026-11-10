# Capítulo 13, Ejercicio 13.4: teoría de valores extremos para las pérdidas del Brent y del S&P 500
import numpy as np, pandas as pd
from scipy import stats
from cap13_funciones import gpd_fit, gpd_var_es, var_es_normal, var_es_t, var_es_hist

b = pd.read_csv("datos/internacional/brent_diario.csv"); s = pd.read_csv("datos/internacional/sp500_diario_yahoo.csv")
series = {"Brent": -100 * np.diff(np.log(b["Price"].to_numpy())), "S&P 500": -100 * np.diff(np.log(s["Adj Close"].to_numpy()))}
for nm, L in series.items():
    n = len(L); print(f"{nm}: n = {n} pérdidas diarias (%)")
    print("  umbral (cuantil)      u     excesos      xi (ee)            beta (ee)")
    for q in (0.90, 0.95, 0.975):
        u = np.quantile(L, q); y = L[L > u] - u; xi, be, ee, _ = gpd_fit(y)
        print(f"  {100 * q:5.1f} %           {u:6.3f}   {len(y):5d}     {xi:6.4f} ({ee[0]:.4f})   {be:6.4f} ({ee[1]:.4f})")
    u = np.quantile(L, 0.95); y = L[L > u] - u; xi, be, _, _ = gpd_fit(y)
    mu, sg = -L.mean(), L.std(ddof=1); e = L - L.mean(); nu = 4 + 6 / ((e**4).mean() / (e**2).mean()**2 - 3)
    print("  con u en el cuantil 95 %:\n  nivel      GPD: VaR    ES     normal: VaR    ES     t: VaR    ES     histórico: VaR    ES")
    for p in (0.99, 0.995, 0.999):
        a = gpd_var_es(u, xi, be, n, len(y), p); bn = var_es_normal(mu, sg, p); bt = var_es_t(mu, sg, nu, p); h = var_es_hist(L, p)
        print(f"  {100 * p:5.1f} %   {a[0]:8.3f} {a[1]:6.3f}   {bn[0]:10.3f} {bn[1]:6.3f}   {bt[0]:7.3f} {bt[1]:6.3f}   {h[0]:12.3f} {h[1]:6.3f}")
    print()
xi_s, be_s, _, _ = gpd_fit((lambda L: L[L > np.quantile(L, .95)] - np.quantile(L, .95))(series["Brent"]))
print(f"comprobación con scipy (Brent, u al 95 %): {stats.genpareto.fit(series['Brent'][series['Brent'] > np.quantile(series['Brent'], .95)] - np.quantile(series['Brent'], .95), floc=0)}")

# ---- Figura: función de media de los excesos y ajuste de la cola (Brent) ----
import estilo_graficos as eg, matplotlib.pyplot as plt
L = series["Brent"]; n = len(L); vs = np.quantile(L, np.linspace(0.80, 0.995, 60))
me = [(L[L > v] - v).mean() for v in vs]
u = np.quantile(L, 0.95); y = L[L > u] - u; xi, be, _, _ = gpd_fit(y)
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
ax[0].plot(vs, me, "o", color=eg.AZUL, ms=3); ax[0].axvline(u, color=eg.LADRILLO, ls="--", lw=1)
ax[0].set_xlabel("umbral $v$ (pérdida, %)"); ax[0].set_ylabel("media de los excesos $e(v)$"); ax[0].set_title("Media de los excesos", fontsize=9)
x = np.sort(L[L > u]); sup = (len(x) - np.arange(len(x))) / n; g = np.linspace(u, x.max(), 200)
ax[1].semilogy(x, sup, ".", color=eg.GRIS, ms=3, label="empírica")
ax[1].semilogy(g, len(y) / n * (1 + xi * (g - u) / be)**(-1 / xi), color=eg.LADRILLO, lw=1.4, label=f"GPD, $\\xi={xi:.3f}$".replace(".", "{,}"))
ax[1].semilogy(g, stats.norm.sf((g - L.mean()) / L.std(ddof=1)), color=eg.AZUL, lw=1.2, ls="--", label="normal")
ax[1].set_ylim(1e-6, 0.06); ax[1].yaxis.set_major_formatter(plt.LogFormatterMathtext()); ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}")); ax[1].set_xlabel("pérdida diaria del Brent (%)"); ax[1].set_ylabel("$\\Pr(L>x)$"); ax[1].legend(fontsize=8)
ax[1].set_title("Cola sobre el umbral del 95 %", fontsize=9)
eg.coma(ax[0]); fig.tight_layout(); fig.savefig("graficos/cap13_ej04_evt.pdf")
