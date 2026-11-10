# Capítulo 15, Ejercicio 15.5: Vasicek y CKLS para la tasa a un mes
import numpy as np
from scipy.stats import norm
from cap15_funciones import cargar

def mco(y, x):
    X = np.column_stack([np.ones(len(y)), x]); b = np.linalg.solve(X.T @ X, X.T @ y); e = y - X @ b
    s2 = (e @ e) / (len(y) - 2); return b, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))), e
f, Y = cargar(); r = Y[:, 0]; n = len(r); Delta = 1 / 12
x, y = r[:-1], r[1:]
(a, b), ee, e = mco(y, x); s = np.sqrt((e @ e) / (n - 1 - 2))
print(f"AR(1) por MCO: r_t = a + b r_{{t-1}} + e_t;  n = {n - 1}")
print(f"  a = {a:.5f} (ee {ee[0]:.5f});  b = {b:.5f} (ee {ee[1]:.5f});  s = {s:.5f}")
kap = -np.log(b) / Delta; th = a / (1 - b); sig = s * np.sqrt(2 * kap / (1 - b**2))
print("Parámetros de Vasicek (Delta = 1/12 año):")
print(f"  kappa = {kap:.4f} por año;  theta = {th:.4f} %;  sigma = {sig:.4f} % por raíz de año")
print(f"  vida media = ln 2 / kappa = {np.log(2) / kap:.3f} años;  desv. est. estacionaria = sigma/sqrt(2 kappa) = {sig / np.sqrt(2 * kap):.4f} %")
print(f"  media muestral = {r.mean():.4f} %;  desv. est. muestral = {r.std(ddof=1):.4f} %;  P(r < 0) bajo Vasicek = {norm.cdf(0, th, sig / np.sqrt(2 * kap)):.4f}")
print("\nEfecto nivel (CKLS): ln e_t^2 = c + 2 gamma ln r_{t-1} + u_t")
(c0, g2), eeg, _ = mco(np.log(e**2), np.log(x)); eeg = eeg[1]
print(f"  2 gamma = {g2:.4f} (ee {eeg:.4f})  ->  gamma = {g2 / 2:.4f};  t(gamma = 0) = {g2 / eeg:.2f};  t(gamma = 0,5) = {(g2 - 1) / eeg:.2f}")
print("\nDesv. est. de e_t por terciles de r_{t-1}:")
q = np.quantile(x, [1 / 3, 2 / 3]); g = np.where(x <= q[0], 0, np.where(x <= q[1], 1, 2))
for k, nom in enumerate(["bajo", "medio", "alto"]):
    print(f"  {nom:<6}  r medio = {x[g == k].mean():6.3f}  desv.est.(e) = {e[g == k].std(ddof=1):.4f}  n = {(g == k).sum()}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.3)); t = np.arange(n)
axs[0].plot(t, r, color=eg.AZUL, lw=0.9, label="$r_t$ (1 mes)"); axs[0].axhline(th, color=eg.LADRILLO, ls="--", lw=1, label=f"$\\hat\\theta$ = {th:.2f}".replace(".", "{,}"))
pos = [k for k in range(n) if f[k].endswith("-01") and int(f[k][:4]) % 10 == 0]
axs[0].set_xticks(pos); axs[0].set_xticklabels([f[k][:4] for k in pos]); axs[0].set_ylabel("% anual"); axs[0].legend(fontsize=8); eg.coma(axs[0])
axs[1].plot(x, e, ".", color=eg.AZUL, ms=3, alpha=0.6); axs[1].axhline(0, color="black", lw=0.5)
axs[1].set_xlabel("$r_{t-1}$ (% anual)"); axs[1].set_ylabel("residuo $\\hat e_t$"); eg.coma(axs[1])
axs[0].set_xticklabels([f[k][:4] for k in pos]); fig.savefig("graficos/cap15_ej05_vasicek.pdf")
