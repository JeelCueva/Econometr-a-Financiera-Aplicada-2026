# Capítulo 8, Ejercicio 8.5: evaluación fuera de muestra de pronósticos de varianza (GARCH, EWMA, ventana móvil)
import numpy as np, pandas as pd, statsmodels.api as sm
from cap08_funciones import garch_fit, garch_s2

b = pd.read_csv("datos/internacional/brent_diario.csv"); r = 100 * np.diff(np.log(b["Price"].to_numpy()))
f = b["Date"].iloc[1:].to_numpy(); dentro = f <= "2015-12-31"; n0 = dentro.sum(); T = len(r)
m = garch_fit(r[dentro], "normal"); p = m["par"]                    # parámetros fijos, estimados hasta 2015
e = r - p["mu"]
h_garch = garch_s2(m["p"], r, m["bc"])                              # el filtro corre sobre toda la muestra
h_ewma = np.empty(T); h_ewma[0] = m["bc"]                           # RiskMetrics: lambda = 0,94
for t in range(1, T): h_ewma[t] = 0.94 * h_ewma[t - 1] + 0.06 * e[t - 1]**2
h_vent = np.full(T, np.nan)                                         # varianza de los 250 días previos
for t in range(250, T): h_vent[t] = (e[t - 250:t]**2).mean()
fuera = np.arange(n0, T); y = e[fuera]**2
print(f"Estimación: {f[0]} a {f[n0 - 1]} ({n0} días); evaluación: {f[n0]} a {f[-1]} ({len(fuera)} días)")
print(f"GARCH estimado: omega = {p['omega']:.4f}, alpha = {p['alpha']:.4f}, beta = {p['beta']:.4f}\n")

def nw_t(d):                                                        # t de Diebold-Mariano con Newey-West
    n = len(d); L = int(np.floor(4 * (n / 100)**(2 / 9))); u = d - d.mean(); s = (u**2).sum() / n
    for j in range(1, L + 1): s += 2 * (1 - j / (L + 1)) * (u[j:] * u[:-j]).sum() / n
    return d.mean() / np.sqrt(s / n)

modelos = {"GARCH": h_garch[fuera], "EWMA": h_ewma[fuera], "Ventana 250": h_vent[fuera]}
qlike = {k: np.log(h) + y / h for k, h in modelos.items()}
print("Modelo        ECM      QLIKE    MZ: a       b      R2")
for nm, h in modelos.items():
    mz = sm.OLS(y, sm.add_constant(h)).fit()
    print(f"{nm:<11} {((y - h)**2).mean():8.2f} {qlike[nm].mean():8.4f} {mz.params[0]:8.3f} {mz.params[1]:7.3f} {mz.rsquared:7.4f}")
print("\nDiebold-Mariano sobre QLIKE (d = pérdida del rival - pérdida del GARCH; t > 0 favorece al GARCH):")
for nm in ["EWMA", "Ventana 250"]:
    d = qlike[nm] - qlike["GARCH"]
    print(f"  GARCH frente a {nm:<11}: media de d = {d.mean():.4f}, t = {nw_t(d):.2f}")

# ---- Figura: los tres pronósticos de volatilidad en 2020 frente a |r_t| ----
import estilo_graficos as eg, matplotlib.pyplot as plt
sel = (f >= "2019-10-01") & (f <= "2021-06-30"); fe = pd.to_datetime(f[sel])
fig, ax = plt.subplots(figsize=(9.0, 3.6))
ax.bar(fe, np.abs(e[sel]), width=1.0, color=eg.GRIS, alpha=0.5, label="$|r_t-\\hat\\mu|$")
ax.plot(fe, np.sqrt(h_garch[sel]), color=eg.LADRILLO, lw=1.1, label="GARCH(1,1)")
ax.plot(fe, np.sqrt(h_ewma[sel]), color=eg.AZUL, lw=1.0, ls="--", label="EWMA ($\\lambda=0{,}94$)")
ax.plot(fe, np.sqrt(h_vent[sel]), color=eg.OLIVA, lw=1.0, ls=":", label="ventana de 250 días")
ax.set_ylim(0, 30); ax.set_ylabel("volatilidad diaria pronosticada (%)"); ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.savefig("graficos/cap08_ej05_fuera_muestra.pdf")
