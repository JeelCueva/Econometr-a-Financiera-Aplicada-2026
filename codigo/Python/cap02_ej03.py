# Capítulo 2, Ejercicio 2.3: gaussianidad agregativa (Brent)
import numpy as np, pandas as pd
from scipy import stats

datos = pd.read_csv("datos/internacional/brent_diario.csv")
r = np.log(datos["Price"]).diff().dropna().to_numpy()

def agregar(x, h):                         # suma de bloques de h días, sin traslape
    n = len(x) // h
    return x[: n * h].reshape(n, h).sum(axis=1)

def resumen(x):
    return {"n": len(x), "desv_est": x.std(), "asimetria": stats.skew(x),
            "curtosis": stats.kurtosis(x, fisher=False), "p_JB": stats.jarque_bera(x).pvalue}

h = [1, 5, 21, 63, 126]
tabla = pd.DataFrame([resumen(agregar(r, hh)) for hh in h], index=[f"h={hh}" for hh in h])
print(tabla.round(4).to_string())
print(f"\nRaíz del tiempo: desv(h=21)/desv(h=1) = {tabla.loc['h=21','desv_est']/tabla.loc['h=1','desv_est']:.3f}"
      f"  vs sqrt(21) = {np.sqrt(21):.3f}")

# ---- Figura: curtosis según el horizonte de agregación ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
hs = [1, 2, 3, 5, 10, 21, 42, 63, 126, 252]
sin_abr20 = np.delete(r, np.where(np.abs(r) > 0.4)[0])   # quita 21 y 22 de abril de 2020
k_all = [stats.kurtosis(agregar(r, x), fisher=False) for x in hs]
k_sin = [stats.kurtosis(agregar(sin_abr20, x), fisher=False) for x in hs]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(hs, k_all, "o-", color=eg.AZUL, label="todos los días")
ax.plot(hs, k_sin, "s--", color=eg.OLIVA, label="sin 21 y 22 de abril de 2020")
ax.axhline(3, color=eg.LADRILLO, ls=":", lw=1.2, label="normal (curtosis = 3)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(hs); ax.set_xticklabels(hs)
ax.set_yticks([2, 3, 5, 10, 20, 50]); ax.set_yticklabels([2, 3, 5, 10, 20, 50])
ax.yaxis.set_minor_formatter(plt.NullFormatter()); ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("horizonte $h$ (días de negociación)"); ax.set_ylabel("curtosis (escala log)")
ax.legend(fontsize=8)
fig.savefig("graficos/cap02_fig_agregacion.pdf")
