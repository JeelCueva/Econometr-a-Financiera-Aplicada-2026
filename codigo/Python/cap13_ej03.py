# Capítulo 13, Ejercicio 13.3: el semáforo de Basilea, año por año
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv("salidas/cap13_var_backtest_Py.csv"); anio = d["fecha"].str[:4].astype(int).to_numpy()
print("Probabilidad binomial de k o menos excepciones en 250 días si la cobertura es correcta (p = 0,01):")
for k in (4, 9): print(f"  P(X <= {k}) = {stats.binom.cdf(k, 250, 0.01):.4f}")
zona = lambda k: "verde" if k <= 4 else ("amarilla" if k <= 9 else "roja")
extra = [0, 0, 0, 0, 0, 0.40, 0.50, 0.65, 0.75, 0.85]                      # factor adicional para k = 0..9
mult = lambda k: 3 + (1 if k >= 10 else extra[k])
nom = ["HS-500", "EWMA-normal", "GARCH-t"]; Z = []
print("\nAño   días   HS-500        EWMA-normal   GARCH-t")
for y in sorted(set(anio)):
    k = anio == y; ex = [int((d["L"][k] > d[m][k]).sum()) for m in nom]; Z.append(ex)
    print(f"{y}  {k.sum():4d}   " + "  ".join(f"{e:2d} {zona(e):<9}" for e in ex))
Z = np.array(Z)
for j, m in enumerate(nom):
    z = Z[:, j]
    print(f"{m:<12} años en verde {(z <= 4).sum():2d}, amarilla {((z >= 5) & (z <= 9)).sum():2d}, roja {(z >= 10).sum():2d}; "
          f"multiplicador medio {np.mean([mult(e) for e in z]):.3f}")

# ---- Figura: excepciones por año y zonas del semáforo ----
import estilo_graficos as eg, matplotlib.pyplot as plt
ys = np.array(sorted(set(anio))); w = 0.27
fig, ax = plt.subplots(figsize=(9.0, 3.6))
ax.axhspan(-0.5, 4.5, color=eg.OLIVA, alpha=0.10); ax.axhspan(4.5, 9.5, color="#C9A227", alpha=0.12); ax.axhspan(9.5, 25, color=eg.LADRILLO, alpha=0.10)
for j, (m, col) in enumerate(zip(nom, [eg.GRIS, eg.AZUL, eg.LADRILLO])):
    ax.bar(ys + (j - 1) * w, Z[:, j], w, color=col, label=m)
ax.set_ylim(0, 23); ax.set_xticks(ys); ax.set_xticklabels([str(y)[2:] for y in ys]); ax.set_xlabel("año (20..)")
ax.set_ylabel("excepciones en el año"); ax.legend(fontsize=8); ax.text(2000.6, 2, "verde", fontsize=8); ax.text(2000.6, 6.8, "amarilla", fontsize=8)
ax.text(2000.6, 15, "roja", fontsize=8); fig.savefig("graficos/cap13_ej03_semaforo.pdf")
