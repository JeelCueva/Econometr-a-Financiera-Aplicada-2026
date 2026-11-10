# Capítulo 18, Ejercicio 18.2: estimación de cuatro cópulas por máxima verosimilitud canónica
import numpy as np
from scipy.stats import kendalltau
from cap18_funciones import indices, pseudo, ajustar, tau_lambda, colas_empiricas, ll_t, dorada

f, x, y = indices(); n = len(x); u, v = pseudo(x), pseudo(y)
A = ajustar(u, v); tau = kendalltau(x, y)[0]
print(f"n = {n}; tau de Kendall empírico = {tau:.4f}\n")
print("cópula      parámetro(s)             log-verosim.     AIC        tau   lambda_inf  lambda_sup")
for nom, k in [("gaussiana", 1), ("t", 2), ("clayton", 1), ("gumbel", 1)]:
    p, ll = A[nom]; tl = tau_lambda(nom, p)
    ps = f"rho = {p[0]:.4f}, nu = {p[1]:.3f}" if nom == "t" else f"{'rho' if nom == 'gaussiana' else 'theta'} = {p:.4f}"
    print(f"{nom:10s}  {ps:22s}  {ll:12.4f}  {-2 * ll + 2 * k:10.2f}  {tl[0]:.4f}  {tl[1]:9.4f}  {tl[2]:9.4f}")
print("\nEstimadores por inversión del tau de Kendall:")
print(f"  gaussiana/t: rho = sin(pi tau / 2) = {np.sin(np.pi * tau / 2):.4f}; Clayton: theta = 2 tau / (1 - tau) = {2 * tau / (1 - tau):.4f}; Gumbel: theta = 1 / (1 - tau) = {1 / (1 - tau):.4f}")
li, ls = colas_empiricas(u, v, 0.01); print(f"Colas empíricas al 1 %: inferior = {li:.4f}; superior = {ls:.4f}")
r, nu = A["t"][0]; lr = 2 * (A["t"][1] - A["gaussiana"][1])
print(f"\nLR de la t frente a la gaussiana (nu -> infinito, en la frontera) = {lr:.2f}")
print("Perfil de la log-verosimilitud de la t:")
for nn in (3, 5, 10, 20, 30):
    print(f"  nu = {nn:2d}: rho = {dorada(lambda rr: ll_t(rr, nn, u, v), -0.99, 0.99)[0]:.4f}; log-verosimilitud = {dorada(lambda rr: ll_t(rr, nn, u, v), -0.99, 0.99)[1]:.4f}")

# ---- Figura: densidades de las cópulas estimadas en escala normal ----
import estilo_graficos as eg, matplotlib.pyplot as plt
from scipy.stats import norm
from cap18_funciones import ll_gauss, ll_clayton, ll_gumbel
g = np.linspace(-3.5, 3.5, 141); X, Y = np.meshgrid(g, g); U, V = norm.cdf(X), norm.cdf(Y)
def dens(fun):
    return np.exp(np.vectorize(lambda a, b: fun(np.array([a]), np.array([b])))(U, V)) * norm.pdf(X) * norm.pdf(Y)
F = {"gaussiana": lambda a, b: ll_gauss(A["gaussiana"][0], a, b), "$t$": lambda a, b: ll_t(r, nu, a, b),
     "Clayton": lambda a, b: ll_clayton(A["clayton"][0], a, b), "Gumbel": lambda a, b: ll_gumbel(A["gumbel"][0], a, b)}
fig, ax = plt.subplots(1, 4, figsize=(10.0, 2.9)); zx, zy = norm.ppf(u), norm.ppf(v)
for a_, (nom, fun) in zip(ax, F.items()):
    a_.plot(zx, zy, ".", ms=0.8, color=eg.GRIS, alpha=0.4)
    a_.contour(X, Y, np.log(dens(fun)), levels=np.arange(-12, 0, 1.5), colors=eg.AZUL, linewidths=0.7)
    a_.set_title(nom, fontsize=9); a_.set_xlim(-3.5, 3.5); a_.set_ylim(-3.5, 3.5); a_.set_aspect("equal"); eg.coma(a_)
fig.tight_layout(); fig.savefig("graficos/cap18_ej02_contornos.pdf")
