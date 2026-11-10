# Capítulo 12, Ejercicio 12.4: cartera de mínima varianza fuera de muestra con covarianzas constante, EWMA y DCC
import numpy as np, pandas as pd
from scipy import stats
from cap12_funciones import panel_diario, etapa1, dcc_fit, dcc_filtro, ewma_cov
from cap08_funciones import garch_s2

f, R = panel_diario(); T, K = R.shape
dentro = f <= "2009-12-31"; fuera = np.flatnonzero(~dentro); n0 = dentro.sum()
Z0, _, mods = etapa1(R[dentro]); m = dcc_fit(Z0)                  # todo se estima con datos hasta 2009
S2 = np.column_stack([garch_s2(mo["p"], R[:, i], mo["bc"]) for i, mo in enumerate(mods)])
Z = (R - np.array([mo["p"][0] for mo in mods])) / np.sqrt(S2)
Rt = dcc_filtro((m["a"], m["b"]), Z, Qb=Z0.T @ Z0 / n0)            # mismos a, b y Qbar de la muestra de estimación
He = ewma_cov(R); Sc = np.cov(R[dentro].T)
pesos = lambda H: (x := np.linalg.solve(H, np.ones(K))) / x.sum()
W = {k: np.zeros((T, K)) for k in ["constante", "EWMA", "DCC"]}; W["iguales"] = np.full((T, K), 1 / 3)
for t in fuera:
    D = np.diag(np.sqrt(S2[t])); W["constante"][t] = pesos(Sc); W["EWMA"][t] = pesos(He[t]); W["DCC"][t] = pesos(D @ Rt[t] @ D)
rp = {k: (w[fuera] * R[fuera]).sum(1) for k, w in W.items()}
print(f"Estimación: {f[0]} a {f[n0 - 1]} ({n0} días); evaluación: {f[fuera[0]]} a {f[-1]} ({len(fuera)} días)")
print(f"DCC estimado: a = {m['a']:.4f}, b = {m['b']:.4f}\n")
print("Estrategia     volat. anual (%)   peso medio S&P  Nasdaq   Brent   rotación diaria media")
for nm, w in W.items():
    w = w[fuera]; rot = np.abs(np.diff(w, axis=0)).sum(1).mean()
    print(f"{nm:<12} {np.sqrt(252) * rp[nm].std(ddof=1):12.3f} {w[:, 0].mean():15.3f} {w[:, 1].mean():8.3f} {w[:, 2].mean():7.3f} {rot:12.4f}")
def nw_t(d):
    n = len(d); L = int(np.floor(4 * (n / 100)**(2 / 9))); u = d - d.mean(); s = (u**2).sum() / n
    for j in range(1, L + 1): s += 2 * (1 - j / (L + 1)) * (u[j:] * u[:-j]).sum() / n
    return d.mean() / np.sqrt(s / n)
print("\nDiebold-Mariano sobre r_p^2 (t > 0: la DCC tiene menor varianza):")
for nm in ["constante", "EWMA", "iguales"]:
    t = nw_t(rp[nm]**2 - rp["DCC"]**2); print(f"  DCC frente a {nm:<10} t = {t:5.2f} (p bilateral = {2 * stats.norm.sf(abs(t)):.4f})")

# ---- Figura: pesos de la cartera de mínima varianza DCC ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fe = pd.to_datetime(f[fuera]); w = W["DCC"][fuera]
fig, ax = plt.subplots(figsize=(9.0, 3.4))
for i, (nm, col, ls) in enumerate([("S&P 500", eg.AZUL, "-"), ("Nasdaq", eg.LADRILLO, "--"), ("Brent", eg.OLIVA, "-")]):
    ax.plot(fe, w[:, i], color=col, ls=ls, lw=0.9, label=nm)
    ax.axhline(W["constante"][fuera[0], i], color=col, ls=":", lw=0.8)
ax.axhline(0, color="black", lw=0.5); ax.set_ylabel("peso en la cartera"); ax.legend(fontsize=8, ncol=3)
eg.coma(ax); ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
fig.savefig("graficos/cap12_ej04_pesos.pdf")
