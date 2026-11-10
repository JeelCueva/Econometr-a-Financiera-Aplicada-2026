# Capítulo 18: gráficas de la teoría (cuatro cópulas con el mismo tau de Kendall = 0,5; cotas de Fréchet-Hoeffding)
import numpy as np
from scipy.stats import norm, t as tdist
import estilo_graficos as eg
import matplotlib.pyplot as plt

rng = np.random.default_rng(18); n = 2000; tau = 0.5; r = np.sin(np.pi * tau / 2); th_c = 2 * tau / (1 - tau); th_g = 1 / (1 - tau)
z1, z2 = rng.standard_normal(n), rng.standard_normal(n); x, y = z1, r * z1 + np.sqrt(1 - r**2) * z2
U = {"gaussiana": (norm.cdf(x), norm.cdf(y))}
w = rng.chisquare(4, n) / 4; U["$t$ con $\\nu=4$"] = (tdist.cdf(x / np.sqrt(w), 4), tdist.cdf(y / np.sqrt(w), 4))
u, q = rng.uniform(size=n), rng.uniform(size=n); U["Clayton"] = (u, ((q**(-th_c / (1 + th_c)) - 1) * u**-th_c + 1)**(-1 / th_c))
a = 1 / th_g; V = rng.uniform(0, np.pi, n); W = rng.exponential(size=n)
S = np.sin(a * V) / np.sin(V)**(1 / a) * (np.sin((1 - a) * V) / W)**((1 - a) / a)
E1, E2 = rng.exponential(size=n), rng.exponential(size=n); U["Gumbel"] = (np.exp(-(E1 / S)**a), np.exp(-(E2 / S)**a))
print(f"tau = {tau}: rho = {r:.4f}; theta Clayton = {th_c:.4f}; theta Gumbel = {th_g:.4f}")
print(f"lambda: t(4) = {2 * tdist.cdf(-np.sqrt(5 * (1 - r) / (1 + r)), 5):.4f}; Clayton inferior = {2**(-1 / th_c):.4f}; Gumbel superior = {2 - 2**(1 / th_g):.4f}")
fig, ax = plt.subplots(2, 4, figsize=(10.0, 5.2))
for k, (nom, (uu, vv)) in enumerate(U.items()):
    ax[0, k].plot(uu, vv, ".", ms=1.2, color=eg.AZUL, alpha=0.5); ax[0, k].set_title(nom, fontsize=9); ax[0, k].set_xlim(0, 1); ax[0, k].set_ylim(0, 1)
    ax[1, k].plot(norm.ppf(uu), norm.ppf(vv), ".", ms=1.2, color=eg.LADRILLO, alpha=0.5); ax[1, k].set_xlim(-4, 4); ax[1, k].set_ylim(-4, 4)
    for a_ in ax[:, k]: a_.set_aspect("equal"); eg.coma(a_)
ax[0, 0].set_ylabel("escala uniforme $(u,v)$"); ax[1, 0].set_ylabel("márgenes $N(0,1)$")
fig.tight_layout(); fig.savefig("graficos/cap18_fig_copulas.pdf")
g = np.linspace(0, 1, 60); G1, G2 = np.meshgrid(g, g)
fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2))
for a_, (Z, t_) in zip(ax, [(np.maximum(G1 + G2 - 1, 0), "cota inferior $W=\\max(u+v-1,0)$"), (G1 * G2, "independencia $\\Pi=uv$"), (np.minimum(G1, G2), "cota superior $M=\\min(u,v)$")]):
    cs = a_.contour(G1, G2, Z, levels=np.arange(0.1, 1.0, 0.1), colors=eg.AZUL, linewidths=0.8); a_.clabel(cs, fontsize=6, fmt=lambda v: f"{v:.1f}".replace(".", ","))
    a_.set_title(t_, fontsize=9); a_.set_aspect("equal"); a_.set_xlabel("$u$"); eg.coma(a_)
ax[0].set_ylabel("$v$"); fig.tight_layout(); fig.savefig("graficos/cap18_fig_frechet.pdf")
