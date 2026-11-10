# Capítulo 9, Ejercicio 9.1: evidencia de asimetría (correlaciones cruzadas y pruebas de sesgo de signo de Engle-Ng)
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from cap09_funciones import ajustar

s = pd.read_csv("datos/internacional/sp500_diario_yahoo.csv")
b = pd.read_csv("datos/internacional/brent_diario.csv")
series = {"S&P 500 (1999-2018)": 100 * np.diff(np.log(s["Adj Close"].to_numpy())),
          "Brent (1987-2026)": 100 * np.diff(np.log(b["Price"].to_numpy()))}
cc = lambda x, y, j: np.corrcoef(x[:len(x) - j], y[j:])[0, 1]          # corr(x_{t-j}, y_t)
corr = {}
for nm, r in series.items():
    e = r - r.mean(); n = len(e)
    corr[nm] = ([cc(e, e**2, j) for j in range(1, 11)], [cc(e**2, e, j) for j in range(1, 11)])
    print(f"{nm}, T = {n}")
    print("  corr(r_{t-j}, r_t^2), j = 1..5:", "".join(f"{v:7.3f}" for v in corr[nm][0][:5]))
    print("  corr(r_{t-j}^2, r_t), j = 1..5:", "".join(f"{v:7.3f}" for v in corr[nm][1][:5]))
    g = ajustar(r, "garch", "c", "normal"); z = g["e"] / np.sqrt(g["s2"])   # residuos del GARCH(1,1) simétrico
    y = z[1:]**2; zl = z[:-1]; Sm = (zl < 0).astype(float)
    X = sm.add_constant(np.column_stack([Sm, Sm * zl, (1 - Sm) * zl]))
    m = sm.OLS(y, X).fit()
    print("  Engle-Ng sobre z_t^2 del GARCH(1,1):")
    for et, c, t in zip(["constante", "sesgo de signo S-", "tamaño negativo S- z", "tamaño positivo S+ z"], m.params, m.tvalues):
        print(f"    {et:<22} {c:8.4f} (t = {t:6.2f})")
    LM = (n - 1) * m.rsquared
    print(f"    prueba conjunta: LM = (T-1) R2 = {LM:.2f}, p = {stats.chi2.sf(LM, 3):.2e}\n")

# ---- Figura: correlaciones cruzadas entre rendimientos y cuadrados ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True); j = np.arange(1, 11); w = 0.38
for a, (nm, (c1, c2)) in zip(ax, corr.items()):
    a.bar(j - w / 2, c1, w, color=eg.LADRILLO, label="corr($r_{t-j}$, $r_t^2$): rendimiento pasado y volatilidad")
    a.bar(j + w / 2, c2, w, color=eg.AZUL, label="corr($r_{t-j}^2$, $r_t$): volatilidad pasada y rendimiento")
    bnd = 1.96 / np.sqrt(len(series[nm]))
    a.axhline(bnd, color=eg.GRIS, ls=":"); a.axhline(-bnd, color=eg.GRIS, ls=":"); a.axhline(0, color="black", lw=0.6)
    a.set_title(nm); a.set_xlabel("rezago $j$ (días)"); eg.coma(a)
ax[0].set_ylabel("correlación")
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.13))
fig.savefig("graficos/cap09_ej01_asimetria.pdf")
