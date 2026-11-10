# Capítulo 18, Ejercicio 18.4: VaR y ES de una cartera S&P 500 / Nasdaq con distintas cópulas
import numpy as np
from scipy.stats import norm, t as tdist, chi2
from cap18_funciones import indices, pseudo, ewma_estandarizar, ajustar, LCG

f, x, y = indices(); lam = 0.94
zx, zy = ewma_estandarizar(x), ewma_estandarizar(y)
hx = np.empty(len(x)); hy = np.empty(len(y)); hx[0], hy[0] = (x[:30]**2).mean(), (y[:30]**2).mean()
for i in range(1, len(x)): hx[i] = lam * hx[i - 1] + (1 - lam) * x[i - 1]**2; hy[i] = lam * hy[i - 1] + (1 - lam) * y[i - 1]**2
sx = np.sqrt(lam * hx[-1] + (1 - lam) * x[-1]**2); sy = np.sqrt(lam * hy[-1] + (1 - lam) * y[-1]**2)
print(f"Volatilidad EWMA pronosticada para el día siguiente a {f[-1]}: S&P 500 = {sx:.4f} %; Nasdaq = {sy:.4f} %")
A = ajustar(pseudo(zx), pseudo(zy)); r, nu = A["t"][0]
print(f"Cópulas de los residuos: gaussiana rho = {A['gaussiana'][0]:.4f}; t rho = {r:.4f}, nu = {nu:.3f}; Clayton theta = {A['clayton'][0]:.4f}; Gumbel theta = {A['gumbel'][0]:.4f}")
N = 20000; g = LCG(2018)
U1, U2, U3, U4 = g.u(N), g.u(N), g.u(N), g.u(N)
S = {}
rg = A["gaussiana"][0]; z1, z2 = norm.ppf(U1), norm.ppf(U2)
S["gaussiana"] = (U1, norm.cdf(rg * z1 + np.sqrt(1 - rg**2) * z2))
W = chi2.ppf(U3, nu) / nu; a1, a2 = z1 / np.sqrt(W), (r * z1 + np.sqrt(1 - r**2) * z2) / np.sqrt(W)
S["t"] = (tdist.cdf(a1, nu), tdist.cdf(a2, nu))
th = A["clayton"][0]; S["Clayton"] = (U1, ((U2**(-th / (1 + th)) - 1) * U1**-th + 1)**(-1 / th))
al = 1 / A["gumbel"][0]; V = np.pi * U1; E = -np.log(U2)
St = np.sin(al * V) / np.sin(V)**(1 / al) * (np.sin((1 - al) * V) / E)**((1 - al) / al)
S["Gumbel"] = (np.exp(-(-np.log(U3) / St)**al), np.exp(-(-np.log(U4) / St)**al))
S["independencia"] = (U1, U2); S["comonotonía"] = (U1, U1)
print(f"\nCartera 50 % S&P 500 y 50 % Nasdaq; {N} simulaciones; márgenes = cuantiles empíricos de los residuos")
print("tau simulado   VaR 99 %   ES 99 %    VaR 99,9 %   cópula")
from scipy.stats import kendalltau
res = {}
for nom, (u, v) in S.items():
    L = -(0.5 * sx * np.quantile(zx, u) + 0.5 * sy * np.quantile(zy, v))
    var1, var2 = np.quantile(L, 0.99), np.quantile(L, 0.999); es = L[L >= var1].mean(); res[nom] = L
    tk = kendalltau(u[:2000], v[:2000])[0] if nom != "comonotonía" else 1.0
    print(f"{tk:12.4f}   {var1:8.4f}  {es:8.4f}  {var2:11.4f}   {nom}")
Lh = -(0.5 * sx * zx + 0.5 * sy * zy); v1 = np.quantile(Lh, 0.99)
print(f"{'':12s}   {v1:8.4f}  {Lh[Lh >= v1].mean():8.4f}  {np.quantile(Lh, 0.999):11.4f}   histórica filtrada (pares observados de residuos, n = {len(Lh)})")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 3.4)); xs = np.linspace(0, 8, 300)
for nom, c, ls in [("gaussiana", eg.AZUL, "-"), ("t", eg.LADRILLO, "--"), ("Clayton", eg.OLIVA, ":"), ("independencia", eg.GRIS, "-.")]:
    ax.plot(xs, [(res[nom] > q).mean() for q in xs], color=c, ls=ls, lw=1.3, label=nom)
ax.set_yscale("log"); ax.set_xlabel("pérdida de la cartera (%)"); ax.set_ylabel("Pr(pérdida > x)"); ax.legend(fontsize=8); eg.coma(ax)
fig.savefig("graficos/cap18_ej04_colas_cartera.pdf")
k = int((Lh >= v1).sum()); cota = 0.5 * sx * np.sort(-zx)[::-1][:k].mean() + 0.5 * sy * np.sort(-zy)[::-1][:k].mean()
print(f"\nCota comonótona exacta del ES histórico (promedio de las {k} mayores pérdidas de cada índice) = {cota:.4f}")
print(f"ES histórico de la cartera = {Lh[Lh >= v1].mean():.4f}; beneficio de diversificación = {cota - Lh[Lh >= v1].mean():.4f} puntos")
