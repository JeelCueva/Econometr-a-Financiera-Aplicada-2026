# Capítulo 9, Ejercicio 9.4: ¿tiene el petróleo efecto apalancamiento? GJR y EGARCH para el Brent
import numpy as np, pandas as pd
from scipy import stats
from cap09_funciones import ajustar, RAIZ_2_PI

b = pd.read_csv("datos/internacional/brent_diario.csv"); r = 100 * np.diff(np.log(b["Price"].to_numpy())); T = len(r)
m = {"GARCH": ajustar(r, "garch", "c", "t"), "GJR": ajustar(r, "gjr", "c", "t"), "EGARCH": ajustar(r, "egarch", "c", "t")}
for nm in ["GJR", "EGARCH"]:
    print(f"{nm}(1,1)-t:")
    tab = pd.DataFrame({"coef": m[nm]["p"], "ee_rob": m[nm]["ee_rob"]}, index=m[nm]["nombres"])
    tab["t_rob"] = (tab.coef / tab.ee_rob).round(2); print(tab.round(4))
LR = 2 * (m["GJR"]["ll"] - m["GARCH"]["ll"])
print(f"\nLR (gamma = 0 en el GJR) = {LR:.2f}, p = {stats.chi2.sf(LR, 1):.4f}")
print("Modelo     k    log-ver        AIC        BIC")
for nm, v in m.items():
    print(f"{nm:<8} {v['k']:3d} {v['ll']:10.2f} {-2 * v['ll'] + 2 * v['k']:10.2f} {-2 * v['ll'] + v['k'] * np.log(T):10.2f}")
p = dict(zip(m["GJR"]["nombres"], m["GJR"]["p"])); q = dict(zip(m["EGARCH"]["nombres"], m["EGARCH"]["p"]))
print(f"GJR: coeficiente de e^2 si e < 0: {p['alpha'] + p['gamma']:.4f}; si e > 0: {p['alpha']:.4f}; "
      f"cociente = {(p['alpha'] + p['gamma']) / p['alpha']:.2f}")
s2 = r.var(ddof=1); x = 2 * np.sqrt(s2)
eg_ = lambda x: np.exp(q["omega"] + q["alpha"] * (np.abs(x / np.sqrt(s2)) - RAIZ_2_PI) + q["gamma"] * x / np.sqrt(s2) + q["beta"] * np.log(s2))
gj = lambda x: p["omega"] + (p["alpha"] + p["gamma"] * (x < 0)) * x**2 + p["beta"] * s2
print(f"choque de +-2 desviaciones ({x:.3f} %): cociente sigma2(-)/sigma2(+): GJR = {gj(-x) / gj(x):.2f}; EGARCH = {eg_(-x) / eg_(x):.2f}")
sub = (b["Date"].iloc[1:] <= "2019-12-31").to_numpy()
mg = ajustar(r[sub], "gjr", "c", "t"); k = mg["nombres"].index("gamma")
print(f"\nGJR-t sin 2020-2026 (hasta 2019, T = {sub.sum()}): gamma = {mg['p'][k]:.4f} "
      f"(ee rob. {mg['ee_rob'][k]:.4f}, t = {mg['p'][k] / mg['ee_rob'][k]:.2f})")

# ---- Figura: curvas de impacto del Brent frente a las del S&P 500, en unidades de desviación estándar ----
import estilo_graficos as eg, matplotlib.pyplot as plt
s = pd.read_csv("datos/internacional/sp500_diario_yahoo.csv"); rs = 100 * np.diff(np.log(s["Adj Close"].to_numpy()))
ps = dict(zip(*[(v := ajustar(rs, "gjr", "c", "t"))["nombres"], v["p"]])); s2s = rs.var(ddof=1)
zs = np.linspace(-4, 4, 400)
rel = lambda f, s2v, x: f(x) / f(0 * x)                    # varianza relativa a la de un choque nulo
gjs = lambda x: ps["omega"] + (ps["alpha"] + ps["gamma"] * (x < 0)) * x**2 + ps["beta"] * s2s
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(zs, rel(gj, s2, zs * np.sqrt(s2)), color=eg.LADRILLO, lw=1.4, label="Brent (GJR-$t$)")
ax.plot(zs, rel(gjs, s2s, zs * np.sqrt(s2s)), color=eg.AZUL, lw=1.4, ls="--", label="S&P 500 (GJR-$t$)")
ax.axvline(0, color="black", lw=0.5); ax.axhline(1, color="black", lw=0.5)
ax.set_xlabel("choque de hoy en desviaciones estándar"); ax.set_ylabel("$\\sigma_t^2$ relativa a un choque nulo")
ax.legend(fontsize=8); eg.coma(ax); fig.savefig("graficos/cap09_ej04_nic_brent.pdf")
