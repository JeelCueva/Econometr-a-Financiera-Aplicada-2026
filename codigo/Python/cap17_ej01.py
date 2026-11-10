# Capítulo 17, Ejercicio 17.1: dos regímenes en la media y la varianza del mercado de EE. UU. (1926-2018)
import numpy as np
from cap17_funciones import mercado, em_media_varianza, ll_mv, hessiano, ergodicas

f, y = mercado(); T = len(y); m, sd = y.mean(), y.std(ddof=0)
print(f"Exceso de rendimiento mensual del mercado, {f[0]} a {f[-1]}, T = {T}")
r = em_media_varianza(y, 2, [m, m], [0.7 * sd, 1.5 * sd], [[0.95, 0.05], [0.05, 0.95]])
print(f"EM: {r['it']} iteraciones; log-verosimilitud = {r['ll']:.4f}")
th = np.r_[r["mu"], r["s"], r["P"][0, 0], r["P"][1, 1]]
H = hessiano(lambda t: ll_mv(t, y, r["rho"]), th); ee = np.sqrt(np.diag(np.linalg.inv(-H)))
print("\nparámetro   estimación     ee")
for nom, v, e in zip(["mu1", "mu2", "sigma1", "sigma2", "p11", "p22"], th, ee): print(f"  {nom:7s} {v:10.4f} {e:9.3f}")
pi = ergodicas(r["P"]); P = r["P"]
print(f"\nprobabilidades estacionarias: {pi[0]:.4f} {pi[1]:.4f}")
print(f"duraciones esperadas (meses): {1 / (1 - P[0, 0]):.2f} {1 / (1 - P[1, 1]):.2f}")
print(f"autocorrelación del régimen, lambda = p11 + p22 - 1 = {P[0, 0] + P[1, 1] - 1:.4f}")
mu, s = r["mu"]; mm = pi @ r["mu"]; vv = pi @ (r["s"]**2 + r["mu"]**2) - mm**2
k4 = (pi @ (3 * r["s"]**4 + 6 * r["s"]**2 * (r["mu"] - mm)**2 + (r["mu"] - mm)**4)) / vv**2
yc = y - y.mean(); kd = (yc**4).mean() / (yc**2).mean()**2
print(f"momentos implícitos: media = {mm:.4f}; desv. est. = {np.sqrt(vv):.4f}; curtosis = {k4:.4f}")
print(f"momentos muestrales: media = {y.mean():.4f}; desv. est. = {y.std(ddof=0):.4f}; curtosis = {kd:.4f}")
lln = -0.5 * T * (np.log(2 * np.pi * sd**2) + 1)
print(f"\nlog-verosimilitud de la normal = {lln:.4f}; diferencia = {r['ll'] - lln:.4f}")
print(f"razón de Sharpe mensual por régimen: {r['mu'][0] / r['s'][0]:.4f} {r['mu'][1] / r['s'][1]:.4f}")
print(f"meses con Pr(régimen 2 | toda la muestra) > 0,5: {(r['xs'][:, 1] > 0.5).sum()} de {T}")

# ---- Figuras ----
import estilo_graficos as eg, matplotlib.pyplot as plt
from scipy.stats import norm
t = np.arange(T); pos = [k for k in range(T) if f[k].endswith("-01") and int(f[k][:4]) % 10 == 0]
fig, ax = plt.subplots(2, 1, figsize=(8.6, 4.8), sharex=True)
ax[0].plot(t, y, color=eg.AZUL, lw=0.5); ax[0].set_ylabel("exceso de rendimiento (%)")
ax[1].plot(t, r["xs"][:, 1], color=eg.LADRILLO, lw=0.8, label="suavizada"); ax[1].plot(t, r["xf"][:, 1], color=eg.GRIS, lw=0.5, alpha=0.8, label="filtrada")
ax[1].set_ylabel("Pr(régimen volátil)"); ax[1].legend(fontsize=8, loc="upper right"); ax[1].set_xticks(pos); ax[1].set_xticklabels([f[k][:4] for k in pos])
for a in ax: eg.coma(a)
ax[1].set_xticklabels([f[k][:4] for k in pos]); fig.savefig("graficos/cap17_ej01_probabilidades.pdf")
fig, ax = plt.subplots(figsize=(6.0, 3.4)); x = np.linspace(-30, 30, 600)
ax.hist(y, bins=np.arange(-30, 40, 1.5), density=True, color=eg.GRIS, alpha=0.5, label="datos")
ax.plot(x, pi[0] * norm.pdf(x, mu, r["s"][0]) + pi[1] * norm.pdf(x, s, r["s"][1]), color=eg.AZUL, lw=1.5, label="mezcla estimada")
ax.plot(x, norm.pdf(x, y.mean(), sd), color=eg.LADRILLO, ls="--", lw=1.2, label="normal"); ax.set_xlim(-30, 30)
ax.set_xlabel("exceso de rendimiento mensual (%)"); ax.set_ylabel("densidad"); ax.legend(fontsize=8); eg.coma(ax)
fig.savefig("graficos/cap17_ej01_densidad.pdf")
