# Capítulo 3, Ejercicio 3.4: betas móviles con ventanas de 60 meses
import numpy as np, pandas as pd

d = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv", index_col="dates")
ventana = 60
x = d["MktRF"]
betas = {}
for i in ["Utils", "Enrgy", "BusEq"]:
    y = d[i] - d["RF"]
    b = (y.rolling(ventana).cov(x) / x.rolling(ventana).var()).dropna()   # beta MCO en la ventana
    betas[i] = b
    print(f"{i:<6} min {b.min():.2f} ({b.idxmin()})  max {b.max():.2f} ({b.idxmax()})  "
          f"último {b.iloc[-1]:.2f} ({b.index[-1]})  desv {b.std():.2f}")

# ---- Figura ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 3.4))
fechas = pd.to_datetime(betas["Utils"].index)
for (i, b), col, ls in zip(betas.items(), [eg.OLIVA, eg.LADRILLO, eg.AZUL], ["-", "--", "-"]):
    ax.plot(fechas, b.to_numpy(), color=col, ls=ls, lw=1.2, label=i)
ax.axhline(1, color=eg.GRIS, ls=":", lw=1)
ax.set_ylabel("beta de 60 meses"); ax.legend(fontsize=8, ncol=3, loc="upper left")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.savefig("graficos/cap03_fig_movil.pdf")
