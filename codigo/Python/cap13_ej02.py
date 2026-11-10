# Capítulo 13, Ejercicio 13.2: prueba retrospectiva del VaR al 99 % del S&P 500 (2001-2018)
import numpy as np, pandas as pd
from scipy import stats
from cap13_funciones import kupiec, christoffersen
from cap08_funciones import garch_fit, garch_s2

s = pd.read_csv("datos/internacional/sp500_diario_yahoo.csv"); r = 100 * np.diff(np.log(s["Adj Close"].to_numpy())); L = -r
f = s["Date"].to_numpy()[1:]; anio = np.array([int(x[:4]) for x in f]); T = len(r); p = 0.99; ev = np.flatnonzero(anio >= 2001)
nom = ["HS-500", "EWMA-normal", "GARCH-t"]; V = np.full((T, 3), np.nan)
for t in ev: V[t, 0] = np.quantile(L[t - 500:t], p)                         # simulación histórica, 500 días
s2 = np.empty(T); s2[0] = r[:30].var(ddof=1)
for t in range(1, T): s2[t] = 0.94 * s2[t - 1] + 0.06 * r[t - 1]**2        # RiskMetrics, media cero
V[ev, 1] = stats.norm.ppf(p) * np.sqrt(s2[ev])
for y in range(2001, 2019):                                                 # GARCH-t reestimado cada enero
    m = garch_fit(r[anio < y], "t"); q = m["par"]; h = garch_s2(m["p"], r, m["bc"]); k = anio == y
    V[k, 2] = -q["mu"] + np.sqrt(h[k]) * np.sqrt((q["nu"] - 2) / q["nu"]) * stats.t.ppf(p, q["nu"])
print(f"Evaluación: {f[ev[0]]} a {f[-1]}, n = {len(ev)} días; excepciones esperadas al 1 %: {0.01 * len(ev):.1f}\n")
print("Modelo         VaR medio  excepciones  tasa (%)  Kupiec LR (p)     Christoffersen LR (p)   cobertura condicional (p)")
X = {}
for j, nm in enumerate(nom):
    x = (L[ev] > V[ev, j]).astype(int); X[nm] = x; ku = kupiec(x, p); ch = christoffersen(x); cc = ku[0] + ch[0]
    print(f"{nm:<12} {V[ev, j].mean():9.3f} {x.sum():10d} {100 * x.mean():10.3f} {ku[0]:8.2f} ({ku[1]:.4f}) "
          f"{ch[0]:10.2f} ({ch[1]:.4f}) {cc:14.2f} ({stats.chi2.sf(cc, 2):.4f})")
k = anio[ev] == 2008
print(f"\nexcepciones en 2008: HS-500 = {X['HS-500'][k].sum()}; EWMA = {X['EWMA-normal'][k].sum()}; "
      f"GARCH-t = {X['GARCH-t'][k].sum()} (de {k.sum()} días)")
pd.DataFrame({"fecha": f[ev], "L": L[ev], **{n: V[ev, j] for j, n in enumerate(nom)}}).to_csv("salidas/cap13_var_backtest_Py.csv", index=False)

# ---- Figura: pérdidas, VaR al 99 % y excepciones en 2007-2009 ----
import estilo_graficos as eg, matplotlib.pyplot as plt
sel = ev[(anio[ev] >= 2007) & (anio[ev] <= 2009)]; fe = pd.to_datetime(f[sel])
fig, ax = plt.subplots(figsize=(9.0, 3.8))
ax.bar(fe, L[sel], width=1.0, color=eg.GRIS, alpha=0.6, label="pérdida diaria")
for j, (col, ls) in enumerate([(eg.OLIVA, ":"), (eg.AZUL, "--"), (eg.LADRILLO, "-")]):
    ax.plot(fe, V[sel, j], color=col, ls=ls, lw=1.1, label=nom[j])
k = L[sel] > V[sel, 2]; ax.plot(fe[k], L[sel][k], "v", color=eg.LADRILLO, ms=4, label="excepción del GARCH-$t$")
ax.set_ylim(-6, 12); ax.set_ylabel("pérdida / VaR (%)"); ax.legend(fontsize=7.5, ncol=3, loc="upper left")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.savefig("graficos/cap13_ej02_backtest.pdf")
