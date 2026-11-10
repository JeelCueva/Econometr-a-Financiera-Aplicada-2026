# Capítulo 16, Ejercicio 16.3: los anuncios del FOMC y el S&P 500 (40 reuniones programadas, 2014-2018)
import numpy as np, pandas as pd
from scipy.stats import norm
from cap16_funciones import rendimientos_normales, pruebas, corrado

s = pd.read_csv("datos/internacional/sp500_diario_yahoo.csv"); P = s["Adj Close"].to_numpy()
f = s["Date"].to_numpy()[1:]; r = 100 * (P[1:] / P[:-1] - 1)
ev = pd.read_csv("datos/internacional/fomc_anuncios_2014_2018.csv")["fecha_anuncio"].to_numpy()
T0 = np.array([int(np.where(f == d)[0][0]) for d in ev]); N = len(T0)
res = [rendimientos_normales(r, r, t0, modelo="media") for t0 in T0]
AR = np.array([x["AR"] for x in res]); V = np.array([np.diag(x["V"]) for x in res]); gl = np.array([x["gl"] for x in res])
print(f"N = {N} anuncios, de {ev[0]} a {ev[-1]}; modelo de rendimiento medio, estimación [-250, -11]\n")
print(" tau    CAAR      J1    Patell    BMP    signo  positivos")
for k, tau in enumerate(range(-5, 6)):
    p = pruebas(AR[:, k], V[:, k], gl)
    print(f" {tau:3d} {p['CAAR']:8.4f} {p['J1']:7.2f} {p['Patell']:7.2f} {p['BMP']:7.2f} {p['signo']:7.2f}   {p['pos']:2d}/{N}")
def caar(t1, t2):
    k = np.arange(t1 + 5, t2 + 6); c = AR[:, k].sum(1); v = np.array([x["V"][np.ix_(k, k)].sum() for x in res])
    sc = c / np.sqrt(v); return c.mean(), c.mean() / np.sqrt(v.sum() / N**2), sc.mean() / (sc.std(ddof=1) / np.sqrt(N))
print("\nventana     CAAR     J1     BMP")
for (a, b) in [(-1, 0), (-1, 1), (-5, -1), (0, 5), (-5, 5)]:
    c, j, bm = caar(a, b); print(f"({a:2d},{b:+d})  {c:8.4f} {j:6.2f} {bm:7.2f}")
A = np.array([np.r_[x["e"], x["AR"][5]] for x in res])
print(f"\nPrueba de rangos de Corrado para el día 0: {corrado(A):.2f}")
# Regresión de días: rendimiento sobre ficticias de día -1 y día 0 del FOMC, 2014-2018
i = np.where((f >= "2014-01-02") & (f <= "2018-12-31"))[0]; d0 = np.isin(i, T0).astype(float); d1 = np.isin(i, T0 - 1).astype(float)
X = np.column_stack([np.ones(len(i)), d1, d0]); y = r[i]; b = np.linalg.solve(X.T @ X, X.T @ y); e = y - X @ b
XtXi = np.linalg.inv(X.T @ X); Vw = XtXi @ (X.T * e**2) @ X @ XtXi
print(f"\nRegresión 2014-2018 (n = {len(i)}): r_t = c + d1 D(-1) + d0 D(0); errores de White (HC0)")
for nom, k in [("constante ", 0), ("día -1    ", 1), ("día 0     ", 2)]: print(f"  {nom} {b[k]:8.4f}  ee {np.sqrt(Vw[k, k]):.4f}  t {b[k] / np.sqrt(Vw[k, k]):6.2f}")
sd0 = r[T0].std(ddof=1); sdo = y[d0 + d1 == 0].std(ddof=1)
print(f"Desv. est. en días de anuncio = {sd0:.4f}; en días sin anuncio (ni víspera) = {sdo:.4f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
tau = np.arange(-5, 6); ca = AR.mean(0).cumsum()
ee = np.array([np.sqrt(np.array([x["V"][:k + 1, :k + 1].sum() for x in res]).sum() / N**2) for k in range(11)])
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.3))
ax[0].bar(tau, AR.mean(0), color=[eg.LADRILLO if t in (-1, 0) else eg.AZUL for t in tau], width=0.7); ax[0].axhline(0, color="black", lw=0.5)
ax[0].set_title("AR medio por día", fontsize=9); ax[0].set_xlabel("$\\tau$"); ax[0].set_ylabel("%")
ax[1].plot(tau, ca, "o-", color=eg.AZUL, ms=3, label="CAAR$(-5,\\tau)$"); ax[1].fill_between(tau, ca - 1.96 * ee, ca + 1.96 * ee, color=eg.AZUL, alpha=0.15, label="banda al 95 % ($J_1$)")
ax[1].axhline(0, color="black", lw=0.5); ax[1].set_title("CAAR de los 40 anuncios", fontsize=9); ax[1].set_xlabel("$\\tau$"); ax[1].legend(fontsize=8)
for a in ax: eg.coma(a)
fig.savefig("graficos/cap16_ej03_fomc.pdf")
