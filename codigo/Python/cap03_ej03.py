# Capítulo 3, Ejercicio 3.3: la recta del mercado de valores (SML) con las 12 industrias
import numpy as np, pandas as pd
import statsmodels.api as sm

d = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv")
industrias = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
              "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]
X = sm.add_constant(d["MktRF"])
beta = pd.Series({i: sm.OLS(d[i] - d["RF"], X).fit().params["MktRF"] for i in industrias})
media = pd.Series({i: 1200 * (d[i] - d["RF"]).mean() for i in industrias})   # % anual
prima = 1200 * d["MktRF"].mean()
print(f"Prima de mercado promedio: {prima:.2f} % anual\n")

sml = sm.OLS(media, sm.add_constant(beta.rename("beta"))).fit()
print(pd.DataFrame({"coef": sml.params, "ee": sml.bse, "t": sml.tvalues, "p": sml.pvalues}).round(3).to_string())
print(f"\nR2 de corte transversal = {sml.rsquared:.3f}")
print(f"Pendiente estimada {sml.params['beta']:.2f} frente a la teórica {prima:.2f}; "
      f"intercepto {sml.params['const']:.2f} frente a 0")

# ---- Figura: SML teórica y estimada ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
g = np.linspace(0, 1.4, 2)
ax.plot(g, prima * g, color=eg.LADRILLO, ls="--", label="SML teórica (CAPM)")
ax.plot(g, sml.params["const"] + sml.params["beta"] * g, color=eg.AZUL, label="SML estimada")
ax.scatter(beta, media, color=eg.AZUL, s=18, zorder=3)
ax.scatter([1], [prima], marker="D", color=eg.LADRILLO, s=30, zorder=3, label="mercado")
desplaz = {"NoDur": (-30, 4), "Enrgy": (4, 4), "Shops": (-30, 4), "Money": (-8, -11), "Manuf": (4, 3),
           "Chems": (4, -9), "Durbl": (4, -9), "Other": (4, -4)}          # posición de cada etiqueta (puntos)
for i in industrias:
    ax.annotate(i, (beta[i], media[i]), xytext=desplaz.get(i, (4, 2)), textcoords="offset points", fontsize=7.5)
ax.set_xlim(0, 1.4); ax.set_ylim(0, 11)
ax.set_xlabel(r"$\hat\beta$"); ax.set_ylabel("exceso de rendimiento promedio (% anual)")
ax.legend(fontsize=8, loc="upper left"); eg.coma(ax)
fig.savefig("graficos/cap03_fig_sml.pdf")
