# Capítulo 20: gráficas de la teoría (contracción de ridge y lasso; sesgo y varianza con polinomios)
import numpy as np
import estilo_graficos as eg
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6)); b = np.linspace(-3, 3, 301); lam = 1.0
ax[0].plot(b, b, color=eg.GRIS, ls=":", lw=1, label="MCO")
ax[0].plot(b, b / (1 + lam), color=eg.AZUL, lw=1.5, label="ridge: $\\hat\\beta/(1+\\lambda)$")
ax[0].plot(b, np.sign(b) * np.maximum(np.abs(b) - lam, 0), color=eg.LADRILLO, lw=1.5, ls="--", label="lasso: $S(\\hat\\beta,\\lambda)$")
ax[0].set_xlabel("estimador de MCO $\\hat\\beta_j$"); ax[0].set_ylabel("estimador contraído"); ax[0].set_title("Regresores ortonormales, $\\lambda=1$", fontsize=9); ax[0].legend(fontsize=8)
rng = np.random.default_rng(20); n = 40; x = np.sort(rng.uniform(-1, 1, n)); fx = np.sin(2.5 * x); y = fx + rng.normal(0, 0.4, n)
xt = rng.uniform(-1, 1, 2000); yt = np.sin(2.5 * xt) + rng.normal(0, 0.4, 2000); ea, ep = [], []
for d in range(0, 13):
    c = np.polyfit(x, y, d); ea.append(((y - np.polyval(c, x))**2).mean()); ep.append(((yt - np.polyval(c, xt))**2).mean())
ax[1].plot(range(13), ea, "o-", color=eg.AZUL, ms=3, label="entrenamiento (n = 40)"); ax[1].plot(range(13), ep, "s--", color=eg.LADRILLO, ms=3, label="datos nuevos")
ax[1].axhline(0.16, color=eg.GRIS, ls=":", lw=1, label="varianza del ruido = 0,16"); ax[1].set_yscale("log"); ax[1].set_xlabel("grado del polinomio"); ax[1].set_ylabel("ECM")
ax[1].set_title("Sesgo y varianza: $y=\\sin(2{,}5x)+\\varepsilon$", fontsize=9); ax[1].legend(fontsize=7)
for a in ax: eg.coma(a)
fig.tight_layout(); fig.savefig("graficos/cap20_fig_teoria.pdf")
print("grado con menor ECM en datos nuevos:", int(np.argmin(ep)), "; ECM:", round(min(ep), 4), "; ECM con grado 12:", round(ep[12], 4))
d = np.loadtxt("datos/internacional/predictores_mensuales_1927_2018.csv", delimiter=",", skiprows=1, usecols=11)
S = d.mean() / d.std(ddof=1)
for R2 in (0.005, 0.01):
    print(f"Campbell-Thompson: S mensual = {S:.4f}; S^2 = {S**2:.5f}; R2 = {R2}: (S^2 + R2)/(1 - R2) = {(S**2 + R2) / (1 - R2):.5f}; aumento proporcional = {((S**2 + R2) / (1 - R2)) / S**2 - 1:.4f}")
