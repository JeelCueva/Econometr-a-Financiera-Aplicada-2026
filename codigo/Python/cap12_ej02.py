# Capítulo 12, Ejercicio 12.2: DCC en dos etapas para el S&P 500 y el Brent
import numpy as np, pandas as pd
from scipy import stats
from cap12_funciones import panel_diario, etapa1, dcc_fit

f, R = panel_diario(); Z, S2, mods = etapa1(R[:, [0, 2]])
print("Etapa 1, GARCH(1,1) normal:")
for nm, m1 in zip(["S&P", "Brent"], mods):
    p = m1["par"]
    print(f"  {nm:<6} mu = {p['mu']:.4f}  omega = {p['omega']:.4f}  alpha = {p['alpha']:.4f}  beta = {p['beta']:.4f}  log-ver = {m1['ll']:.2f}")
m = dcc_fit(Z); LR = 2 * (m["ll"] - m["ll_ccc"])
print(f"\nEtapa 2, DCC(1,1): a = {m['a']:.4f} (ee {m['ee'][0]:.4f}), b = {m['b']:.4f} (ee {m['ee'][1]:.4f}); "
      f"a + b = {m['a'] + m['b']:.4f}; vida media = {np.log(0.5) / np.log(m['a'] + m['b']):.0f} días")
print(f"log-verosimilitud de correlación: DCC = {m['ll']:.2f}; CCC = {m['ll_ccc']:.2f}; LR = {LR:.2f} (p = {stats.chi2.sf(LR, 2):.2e}, 2 g.l.)")
print(f"correlación constante (CCC) de los residuos estandarizados = {m['Rc'][0, 1]:.4f}")
rho = m["R"][:, 0, 1]
print(f"correlación DCC: mín. {rho.min():.3f} ({f[rho.argmin()]}), máx. {rho.max():.3f} ({f[rho.argmax()]})")
print("promedio anual de la correlación DCC:")
anio = pd.Series(f).str[:4].astype(int).to_numpy()
for y in [2000, 2003, 2006, 2008, 2009, 2012, 2015, 2018]: print(f"  {y}: {rho[anio == y].mean():6.3f}")

# ---- Figura: correlación DCC, CCC y volatilidades condicionales ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fe = pd.to_datetime(f)
fig, ax = plt.subplots(2, 1, figsize=(9.0, 5.0), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
ax[0].plot(fe, rho, color=eg.AZUL, lw=1.0, label="DCC"); ax[0].axhline(m["Rc"][0, 1], color=eg.LADRILLO, ls="--", lw=1, label="CCC")
ax[0].axhline(0, color="black", lw=0.5); ax[0].set_ylabel("correlación S&P–Brent"); ax[0].legend(fontsize=8)
ax[1].plot(fe, np.sqrt(252 * S2[:, 0]), color=eg.GRIS, lw=0.8, label="S&P 500")
ax[1].plot(fe, np.sqrt(252 * S2[:, 1]), color=eg.OLIVA, lw=0.8, label="Brent"); ax[1].set_ylabel("volatilidad anual (%)")
ax[1].legend(fontsize=8)
for a in ax: eg.coma(a)
ax[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
fig.savefig("graficos/cap12_ej02_dcc.pdf")
