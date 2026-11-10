# Capítulo 18, Ejercicio 18.1: medidas de dependencia del S&P 500 y el Nasdaq (1999-2018)
import numpy as np
from scipy.stats import kendalltau, norm
from cap18_funciones import indices, pseudo, colas_empiricas, phi2_diag

f, x, y = indices(); n = len(x); u, v = pseudo(x), pseudo(y)
print(f"Rendimientos logarítmicos diarios (%), {f[0]} a {f[-1]}; n = {n}")
for nom, z in [("S&P 500", x), ("Nasdaq ", y)]:
    zc = z - z.mean(); print(f"  {nom}: media = {z.mean():.4f}; desv. est. = {z.std(ddof=1):.4f}; asimetría = {(zc**3).mean() / (zc**2).mean()**1.5:.4f}; curtosis = {(zc**4).mean() / (zc**2).mean()**2:.4f}")
rp = np.corrcoef(x, y)[0, 1]; tau = kendalltau(x, y)[0]; rs = np.corrcoef(u, v)[0, 1]
print(f"\nPearson = {rp:.4f}; tau de Kendall = {tau:.4f}; rho de Spearman = {rs:.4f}")
print(f"Spearman calculado con las pseudo-observaciones = Pearson de los rangos: {np.corrcoef(u, v)[0, 1]:.4f}")
rg = np.sin(np.pi * tau / 2)
print(f"rho de la cópula gaussiana con el mismo tau: sin(pi tau / 2) = {rg:.4f}; Pearson de las normales de rangos = {np.corrcoef(norm.ppf(u), norm.ppf(v))[0, 1]:.4f}")
print("\nDependencia en las colas: Pr(V <= q | U <= q) y Pr(V > 1 - q | U > 1 - q)")
print("     q     inferior   superior   gaussiana con rho = sin(pi tau / 2)")
for q in (0.10, 0.05, 0.01):
    li, ls = colas_empiricas(u, v, q); g = phi2_diag(norm.ppf(q), rg) / q
    print(f"  {q:5.2f}   {li:8.4f}   {ls:8.4f}   {g:8.4f}")
print(f"\nPeores 5 días del S&P 500 y el rango del Nasdaq ese día:")
for i in np.argsort(x)[:5]: print(f"  {f[i]}: S&P 500 = {x[i]:8.4f} %; Nasdaq = {y[i]:8.4f} % (pseudo-observación {v[i]:.4f})")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.0, 4.2))
ax[0].plot(x, y, ".", ms=1.5, color=eg.AZUL, alpha=0.4); ax[0].set_xlabel("S&P 500 (%)"); ax[0].set_ylabel("Nasdaq (%)"); ax[0].set_title("Rendimientos", fontsize=9)
ax[1].plot(u, v, ".", ms=1.5, color=eg.LADRILLO, alpha=0.4); ax[1].set_xlabel("$u$ (S&P 500)"); ax[1].set_ylabel("$v$ (Nasdaq)"); ax[1].set_title("Pseudo-observaciones (la cópula empírica)", fontsize=9)
for a in ax: a.set_aspect("equal", adjustable="box"); eg.coma(a)
fig.tight_layout(); fig.savefig("graficos/cap18_ej01_dispersion.pdf")
