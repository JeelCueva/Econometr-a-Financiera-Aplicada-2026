# Capítulo 2, Ejercicio 2.5: índice de cola de Hill para pérdidas y ganancias del Brent
import numpy as np, pandas as pd

datos = pd.read_csv("datos/internacional/brent_diario.csv")
r = np.log(datos["Price"]).diff().dropna().to_numpy()

def hill(x, k):                            # x > 0; usa las k mayores observaciones
    xs = np.sort(x[x > 0])[::-1]
    return k / np.sum(np.log(xs[:k] / xs[k]))

k = np.array([50, 100, 200, 400])
tabla = pd.DataFrame({"k": k, "pct_muestra": 100 * k / len(r),
                      "alfa_perdidas": [hill(-r, kk) for kk in k],
                      "alfa_ganancias": [hill(r, kk) for kk in k]})
tabla["ee_perdidas"] = tabla.alfa_perdidas / np.sqrt(k)
tabla["ee_ganancias"] = tabla.alfa_ganancias / np.sqrt(k)
print(tabla.round(3).to_string(index=False))

# ---- Figura: gráfico de Hill ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
ks = np.arange(20, 901)
fig, ax = plt.subplots(figsize=(6.2, 3.3))
ax.plot(ks, [hill(-r, kk) for kk in ks], color=eg.LADRILLO, label="pérdidas ($-r_t$)")
ax.plot(ks, [hill(r, kk) for kk in ks], color=eg.AZUL, ls="--", label="ganancias ($r_t$)")
ax.axhline(4, color=eg.GRIS, ls=":", lw=1, label=r"$\alpha=4$ (cuarto momento)")
ax.axhline(2, color=eg.GRIS, ls="-.", lw=1, label=r"$\alpha=2$ (varianza)")
ax.set_xlabel("número de observaciones de la cola $k$"); ax.set_ylabel(r"$\hat\alpha$ de Hill")
ax.set_ylim(0, 6); ax.legend(fontsize=8, ncol=2); eg.coma(ax)
fig.savefig("graficos/cap02_fig_hill.pdf")
