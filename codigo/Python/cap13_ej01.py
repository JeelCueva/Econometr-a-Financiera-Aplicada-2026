# Capítulo 13, Ejercicio 13.1: VaR y ES incondicionales a un día: normal, t de Student e histórico
import numpy as np, pandas as pd
from cap13_funciones import var_es_normal, var_es_t, var_es_hist

s = pd.read_csv("datos/internacional/sp500_diario_yahoo.csv"); b = pd.read_csv("datos/internacional/brent_diario.csv")
series = {"S&P 500 (1999-2018)": -100 * np.diff(np.log(s["Adj Close"].to_numpy())),
          "Brent (1987-2026)": -100 * np.diff(np.log(b["Price"].to_numpy()))}
for nm, L in series.items():
    mu, sg = -L.mean(), L.std(ddof=1); e = L - L.mean(); k = (e**4).mean() / (e**2).mean()**2
    nu = 4 + 6 / (k - 3)                                  # t por momentos: curtosis = 3 + 6/(nu - 4)
    print(f"{nm}: T = {len(L)}, media de r = {mu:.4f} %, desv. est. = {sg:.4f} %, curtosis = {k:.2f} -> nu = {nu:.3f}")
    print("  nivel   modelo       VaR      ES   frecuencia de L > VaR")
    for p in (0.95, 0.99):
        res = {"normal": var_es_normal(mu, sg, p), "t": var_es_t(mu, sg, nu, p), "histórico": var_es_hist(L, p)}
        for m, (v, es) in res.items():
            print(f"  {100 * p:4.0f} %   {m:<10} {v:6.3f}  {es:6.3f}   {100 * (L > v).mean():6.3f} %")
    print()

# ---- Figura: cola izquierda empírica frente a las ajustadas (S&P 500) ----
import estilo_graficos as eg, matplotlib.pyplot as plt
from scipy import stats
L = series["S&P 500 (1999-2018)"]; mu, sg = -L.mean(), L.std(ddof=1); e = L - L.mean(); nu = 4 + 6 / ((e**4).mean() / (e**2).mean()**2 - 3)
x = np.sort(L); sup = 1 - np.arange(1, len(x) + 1) / (len(x) + 1); g = np.linspace(1, 10, 200)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.semilogy(x[x > 1], sup[x > 1], ".", color=eg.GRIS, ms=3, label="empírica")
ax.semilogy(g, stats.norm.sf((g + mu) / sg), color=eg.AZUL, lw=1.4, label="normal")
ax.semilogy(g, stats.t.sf((g + mu) / (sg * np.sqrt((nu - 2) / nu)), nu), color=eg.LADRILLO, lw=1.4, ls="--", label=f"$t$ con $\\nu={nu:.2f}$".replace(".", "{,}"))
ax.axhline(0.01, color="black", lw=0.5, ls=":"); ax.set_ylim(1e-5, 0.3)
ax.set_xlabel("pérdida diaria del S&P 500 (%)"); ax.set_ylabel("$\\Pr(L > x)$"); ax.legend(fontsize=8); eg.coma(ax)
ax.yaxis.set_major_formatter(plt.LogFormatterMathtext()); fig.savefig("graficos/cap13_ej01_colas.pdf")
