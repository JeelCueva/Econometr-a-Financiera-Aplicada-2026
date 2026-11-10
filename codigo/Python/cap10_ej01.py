# Capítulo 10, Ejercicio 10.1: VAR para el S&P 500 y el VIX: selección del orden, estimación y estabilidad
import numpy as np, pandas as pd
from cap10_funciones import seleccion, var_mco, companera
from statsmodels.tsa.api import VAR                       # solo para comprobar

s = pd.read_csv("datos/internacional/sp500_diario_yahoo.csv"); v = pd.read_csv("datos/internacional/vix_diario.csv")
d = s[["Date", "Adj Close"]].merge(v[["DATE", "CLOSE"]], left_on="Date", right_on="DATE")
Y = np.column_stack([100 * np.diff(np.log(d["Adj Close"])), 100 * np.diff(np.log(d["CLOSE"]))])
print(f"T = {len(Y)} días ({d['Date'].iloc[1]} a {d['Date'].iloc[-1]}); medias: r = {Y[:, 0].mean():.4f}, "
      f"dv = {Y[:, 1].mean():.4f}; desv. est.: {Y[:, 0].std(ddof=1):.4f} y {Y[:, 1].std(ddof=1):.4f}\n")
ic = seleccion(Y, 10)
print("  p      AIC      BIC       HQ")
for f in ic: print(f"{int(f[0]):3d} {f[1]:8.4f} {f[2]:8.4f} {f[3]:8.4f}")
print(f"orden elegido: AIC = {int(ic[ic[:, 1].argmin(), 0])}, BIC = {int(ic[ic[:, 2].argmin(), 0])}, "
      f"HQ = {int(ic[ic[:, 3].argmin(), 0])}\n")
m = var_mco(Y, 2); XtXi = np.linalg.inv(m["X"].T @ m["X"])
nom = ["constante", "r(t-1)", "dv(t-1)", "r(t-2)", "dv(t-2)"]
for k, et in enumerate(["r", "dv"]):
    ee = np.sqrt(np.diag(XtXi) * m["S"][k, k]); print(f"Ecuación de {et}:")
    for i in range(5): print(f"  {nom[i]:<10} {m['B'][i, k]:9.4f}  (ee {ee[i]:.4f}, t = {m['B'][i, k] / ee[i]:6.2f})")
print("\nMatriz de covarianzas de los residuos:"); print(np.round(m["S"], 4))
print(f"correlación de los residuos = {m['S'][0, 1] / np.sqrt(m['S'][0, 0] * m['S'][1, 1]):.4f}")
print("módulos de los autovalores de la matriz compañera:", " ".join(f"{x:.4f}" for x in np.abs(np.linalg.eigvals(companera(m["A"])))))
r = VAR(Y).fit(2)
print(f"\ncomprobación con statsmodels: coeficientes iguales = {np.allclose(r.params, m['B'])}; "
      f"Sigma igual = {np.allclose(r.sigma_u, m['S'])}; orden BIC = {VAR(Y).select_order(10).bic}")

# ---- Figura: rendimiento del S&P 500 frente al cambio del VIX el mismo día ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.4, 3.8))
ax.plot(Y[:, 0], Y[:, 1], ".", color=eg.AZUL, ms=1.5, alpha=0.5)
b = np.polyfit(Y[:, 0], Y[:, 1], 1); xs = np.linspace(-9.5, 11.5, 50)
ax.plot(xs, np.polyval(b, xs), color=eg.LADRILLO, lw=1.2, label=f"recta MCO: pendiente {b[0]:.2f}".replace(".", ","))
ax.set_xlabel("rendimiento del S&P 500 (%)"); ax.set_ylabel("cambio logarítmico del VIX (%)")
ax.legend(fontsize=8); eg.coma(ax); fig.savefig("graficos/cap10_ej01_dispersion.pdf")
