# Capítulo 15, Ejercicio 15.4: Nelson-Siegel dinámico (Diebold y Li)
import numpy as np
from cap15_funciones import cargar, ns_ajuste, ns_lambda, ns_cargas, TAU

def acf(x, k): x = x - x.mean(); return (x[k:] * x[:-k]).sum() / (x * x).sum()
f, Y = cargar()
malla = np.round(np.arange(0.010, 0.2000001, 0.0005), 4)
lam_opt, sse = ns_lambda(Y, TAU, malla)
print(f"lambda común (malla 0,010 a 0,200, paso 0,0005): {lam_opt:.4f}  -> curvatura máxima en tau = {1.793 / lam_opt:.1f} meses")
for lam in (lam_opt, 0.0609):
    B, A, recm = ns_ajuste(Y, TAU, lam)
    print(f"\nlambda = {lam:.4f}: RECM global = {recm:.4f} puntos porcentuales")
    print("  RECM por vencimiento:", "  ".join(f"{m}:{v:.3f}" for m, v in zip(TAU, np.sqrt(((Y - A)**2).mean(0)))), "")
    print("  factor     media    desv.est.  rho(1)   rho(12)")
    for j in range(3):
        b = B[:, j]; print(f"  beta{j}  {b.mean():9.4f} {b.std(ddof=1):9.4f} {acf(b, 1):8.4f} {acf(b, 12):8.4f}")
B, A, recm = ns_ajuste(Y, TAU, 0.0609)
niv = Y[:, 9]; pen = Y[:, 9] - Y[:, 2]; cur = 2 * Y[:, 7] - Y[:, 2] - Y[:, 9]
c = lambda u, v: np.corrcoef(u, v)[0, 1]
print("\nCorrelaciones (lambda = 0,0609) con los factores empíricos:")
print(f"  corr(beta0, y120)                 = {c(B[:, 0], niv):.4f}")
print(f"  corr(-beta1, y120 - y3)           = {c(-B[:, 1], pen):.4f}")
print(f"  corr(beta2, 2 y36 - y3 - y120)    = {c(B[:, 2], cur):.4f}")
print("\nFactores y ajuste en fechas elegidas (lambda = 0,0609):")
fechas = ["1950-01", "1981-09", "1991-02"]
for d in fechas:
    i = np.where(f == d)[0][0]
    print(f"  {d}: beta0 = {B[i, 0]:.3f}  beta1 = {B[i, 1]:.3f}  beta2 = {B[i, 2]:.3f}  RECM del mes = {np.sqrt(((Y[i] - A[i])**2).mean()):.3f}")

# ---- Figuras ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=False); tt = np.linspace(1, 120, 300)
for ax, d in zip(axs, fechas):
    i = np.where(f == d)[0][0]
    ax.plot(TAU, Y[i], "o", color=eg.AZUL, ms=4, label="observada")
    ax.plot(tt, ns_cargas(tt, 0.0609) @ B[i], color=eg.LADRILLO, lw=1.3, label="Nelson-Siegel")
    ax.set_title(d, fontsize=9); ax.set_xlabel("vencimiento (meses)"); eg.coma(ax)
axs[0].set_ylabel("tasa (% anual)"); axs[0].legend(fontsize=7); fig.savefig("graficos/cap15_ej04_ajustes.pdf")
fig, axs = plt.subplots(3, 1, figsize=(6.4, 5.6), sharex=True); t = np.arange(len(f))
pares = [(B[:, 0], niv, "$\\hat\\beta_0$", "$y_{120}$"), (-B[:, 1], pen, "$-\\hat\\beta_1$", "$y_{120}-y_3$"), (B[:, 2], cur, "$\\hat\\beta_2$", "$2y_{36}-y_3-y_{120}$")]
for ax, (u, v, lu, lv) in zip(axs, pares):
    ax.plot(t, u, color=eg.AZUL, lw=0.9, label=lu); ax.plot(t, v, color=eg.LADRILLO, lw=0.9, label=lv); ax.legend(fontsize=7, loc="upper left"); eg.coma(ax)
pos = [k for k in range(len(f)) if f[k].endswith("-01") and int(f[k][:4]) % 10 == 0]
axs[2].set_xticks(pos); axs[2].set_xticklabels([f[k][:4] for k in pos]); fig.savefig("graficos/cap15_ej04_factores.pdf")
fig, ax = plt.subplots(figsize=(5.6, 3.0)); ax.plot(malla, sse, color=eg.AZUL, lw=1.2)
ax.axvline(lam_opt, color=eg.LADRILLO, ls="--", lw=1, label=f"mínimo: {lam_opt:.4f}".replace(".", ","))
ax.axvline(0.0609, color=eg.GRIS, ls=":", lw=1, label="Diebold-Li: 0,0609"); ax.set_xlabel("$\\lambda$"); ax.set_ylabel("SCE total"); ax.legend(fontsize=8); eg.coma(ax)
fig.savefig("graficos/cap15_ej04_lambda.pdf")
