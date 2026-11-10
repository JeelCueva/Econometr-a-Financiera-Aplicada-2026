# Capítulo 4, Ejercicio 4.1: heterocedasticidad en el modelo de mercado de Utils
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

d = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv")
y = (d["Utils"] - d["RF"]).to_numpy(); x = d["MktRF"].to_numpy(); T = len(y)
X = np.column_stack([np.ones(T), x]); k = X.shape[1]
XtXi = np.linalg.inv(X.T @ X)
b = XtXi @ X.T @ y; e = y - X @ b

def lm_prueba(Z):                           # LM = T * R2 de e^2 sobre Z
    m = sm.OLS(e**2, sm.add_constant(Z)).fit()
    LM = T * m.rsquared; q = Z.shape[1] if Z.ndim > 1 else 1
    return np.round([LM, q, stats.chi2.sf(LM, q)], 4)
print("Breusch-Pagan:", lm_prueba(x))
print("White:        ", lm_prueba(np.column_stack([x, x**2])), "\n")

s2 = e @ e / (T - k)
V = {"clasico": s2 * XtXi}
V["HC0"] = XtXi @ ((X * e[:, None]).T @ (X * e[:, None])) @ XtXi
V["HC1"] = V["HC0"] * T / (T - k)
ee = pd.DataFrame({n: np.sqrt(np.diag(v)) for n, v in V.items()}, index=["alfa", "beta"])
print(ee.round(5).to_string())
print("\nt(alfa = 0):", (b[0] / ee.loc["alfa"]).round(3).to_numpy())
print("t(beta = 1):", ((b[1] - 1) / ee.loc["beta"]).round(3).to_numpy())
print(f"Cociente ee(beta) HC1/clásico = {ee.loc['beta','HC1']/ee.loc['beta','clasico']:.3f}")
print("\nComprobación con statsmodels (HC1):", sm.OLS(y, X).fit(cov_type="HC1").bse.round(5))

# ---- Figura: residuos al cuadrado frente al mercado ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.8, 3.6))
ax.scatter(100 * x, 1e4 * e**2, s=6, color=eg.AZUL, alpha=0.5, label="$\\hat\\varepsilon_t^2$")
c = np.polyfit(x, e**2, 2); g = np.linspace(x.min(), x.max(), 200)
ax.plot(100 * g, 1e4 * np.polyval(c, g), color=eg.LADRILLO, lw=1.6, label="ajuste cuadrático (regresión de White)")
ax.set_xlabel("exceso de rendimiento del mercado (%)"); ax.set_ylabel("residuo al cuadrado (%²)")
ax.set_ylim(0, 60); ax.legend(fontsize=8); eg.coma(ax)
fig.savefig("graficos/cap04_fig_white.pdf")
