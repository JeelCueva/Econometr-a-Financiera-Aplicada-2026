# Capítulo 18, Ejercicio 18.3: la cópula de los residuos estandarizados (EWMA) frente a la de los rendimientos
import numpy as np
from scipy.stats import kendalltau
from cap18_funciones import indices, pseudo, ewma_estandarizar, ajustar, tau_lambda, colas_empiricas

f, x, y = indices(); zx, zy = ewma_estandarizar(x), ewma_estandarizar(y)
k = lambda z: ((z - z.mean())**4).mean() / ((z - z.mean())**2).mean()**2
print(f"Residuos estandarizados con EWMA (lambda = 0,94): curtosis S&P 500 = {k(zx):.4f}; Nasdaq = {k(zy):.4f}")
print(f"  desv. est. = {zx.std(ddof=1):.4f} y {zy.std(ddof=1):.4f}; Pearson = {np.corrcoef(zx, zy)[0, 1]:.4f}; tau de Kendall = {kendalltau(zx, zy)[0]:.4f}")
R = {"rendimientos": (pseudo(x), pseudo(y)), "residuos EWMA": (pseudo(zx), pseudo(zy))}
print("\n                 lambda_inf(q)          lambda_sup(q)")
print("                 10 %    5 %    1 %    10 %    5 %    1 %")
for nom, (u, v) in R.items():
    L = [colas_empiricas(u, v, q) for q in (0.10, 0.05, 0.01)]
    print(f"  {nom:13s}  " + " ".join(f"{l[0]:.3f}" for l in L) + "  " + " ".join(f"{l[1]:.3f}" for l in L))
print("\ndatos           cópula      parámetro(s)               log-verosim.      AIC    lambda_inf lambda_sup")
for nom, (u, v) in R.items():
    A = ajustar(u, v)
    for c, kk in [("gaussiana", 1), ("t", 2), ("clayton", 1), ("gumbel", 1)]:
        p, ll = A[c]; tl = tau_lambda(c, p)
        ps = f"rho = {p[0]:.4f}, nu = {p[1]:.3f}" if c == "t" else f"{'rho' if c == 'gaussiana' else 'theta'} = {p:.4f}"
        print(f"{nom:14s}  {c:10s}  {ps:24s}  {ll:12.4f}  {-2 * ll + 2 * kk:10.2f}  {tl[1]:8.4f}  {tl[2]:8.4f}")
    if nom == "residuos EWMA": At = A

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
qs = np.linspace(0.01, 0.20, 39); fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
for a, (nom, (u, v)) in zip(ax, R.items()):
    a.plot(qs, [colas_empiricas(u, v, q)[0] for q in qs], color=eg.AZUL, lw=1.4, label="cola inferior")
    a.plot(qs, [colas_empiricas(u, v, q)[1] for q in qs], color=eg.LADRILLO, lw=1.4, ls="--", label="cola superior")
    a.set_title(nom.capitalize(), fontsize=9); a.set_xlabel("$q$"); eg.coma(a)
ax[0].set_ylabel("$\\hat\\lambda(q)$"); ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig("graficos/cap18_ej03_colas.pdf")
