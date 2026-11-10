# Capítulo 3, Ejercicio 3.2: modelo de mercado para las 12 industrias
import numpy as np, pandas as pd
import statsmodels.api as sm

d = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv")
industrias = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
              "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]
X = sm.add_constant(d["MktRF"])
filas = {}
for i in industrias:
    m = sm.OLS(d[i] - d["RF"], X).fit()
    filas[i] = {"alfa_anual_pct": 1200 * m.params["const"], "t_alfa": m.tvalues["const"],
                "beta": m.params["MktRF"], "ee_beta": m.bse["MktRF"],
                "t_beta1": (m.params["MktRF"] - 1) / m.bse["MktRF"], "R2": m.rsquared}
tabla = pd.DataFrame(filas).T.sort_values("beta")
print(tabla.round(3).to_string())
print(f"\nIndustrias con |t(alfa)| > 1,96: {(tabla.t_alfa.abs() > 1.96).sum()} de {len(tabla)}")

# ---- Figura: betas y alfas con intervalos al 95 % ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
pos = np.arange(len(tabla))
ax[0].errorbar(tabla.beta, pos, xerr=1.96 * tabla.ee_beta, fmt="o", color=eg.AZUL, ms=4, capsize=2)
ax[0].axvline(1, color=eg.LADRILLO, ls="--", lw=1); ax[0].set_xlabel(r"$\hat\beta$ (IC 95 %)")
ee_alfa = tabla.alfa_anual_pct / tabla.t_alfa
ax[1].errorbar(tabla.alfa_anual_pct, pos, xerr=1.96 * ee_alfa, fmt="s", color=eg.OLIVA, ms=4, capsize=2)
ax[1].axvline(0, color=eg.LADRILLO, ls="--", lw=1); ax[1].set_xlabel(r"$\hat\alpha$ anual en % (IC 95 %)")
from matplotlib.ticker import FuncFormatter
for a in ax: a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
ax[0].set_yticks(pos); ax[0].set_yticklabels(tabla.index)
fig.savefig("graficos/cap03_fig_betas.pdf")
