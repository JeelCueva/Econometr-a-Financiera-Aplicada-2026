# Capítulo 4, Ejercicio 4.3: autocorrelaciones del Brent con bandas robustas
import numpy as np, pandas as pd
from scipy import stats

datos = pd.read_csv("datos/internacional/brent_diario.csv")
r = np.log(datos["Price"]).diff().dropna().to_numpy(); r = r - r.mean(); T = len(r)
filas = []
for j in range(1, 11):
    a, b = r[j:], r[:-j]
    rho = np.sum(a * b) / np.sum(r**2)
    v = np.sum(a**2 * b**2) / np.sum(r**2)**2   # varianza robusta de rho_j
    filas.append({"rezago": j, "rho": rho, "banda_clasica": 1.96 / np.sqrt(T), "banda_robusta": 1.96 * np.sqrt(v),
                  "t_clasico": rho * np.sqrt(T), "t_robusto": rho / np.sqrt(v)})
res = pd.DataFrame(filas)
print(res.round(4).to_string(index=False))
Q_lb = T * (T + 2) * np.sum(res.rho**2 / (T - res.rezago))
Q_rob = np.sum(res.t_robusto**2)
print(f"\nLjung-Box Q(10) = {Q_lb:.2f}  p = {stats.chi2.sf(Q_lb, 10):.4f}")
print(f"Robusta  Q*(10) = {Q_rob:.2f}  p = {stats.chi2.sf(Q_rob, 10):.4f}")

# ---- Figura ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.bar(res.rezago, res.rho, color=eg.AZUL, width=0.5, label=r"$\hat\rho_j$")
ax.plot(res.rezago, res.banda_clasica, color=eg.GRIS, ls="--", label="banda de Bartlett (i.i.d.)")
ax.plot(res.rezago, -res.banda_clasica, color=eg.GRIS, ls="--")
ax.plot(res.rezago, res.banda_robusta, color=eg.LADRILLO, marker="o", ms=3, label="banda robusta")
ax.plot(res.rezago, -res.banda_robusta, color=eg.LADRILLO, marker="o", ms=3)
ax.axhline(0, color="black", lw=0.6); ax.set_xticks(res.rezago)
ax.set_xlabel("rezago (días)"); ax.set_ylabel("autocorrelación"); ax.legend(fontsize=8, ncol=3, loc="upper center")
ax.set_ylim(-0.11, 0.13); eg.coma(ax)
fig.savefig("graficos/cap04_fig_acfrobusta.pdf")
