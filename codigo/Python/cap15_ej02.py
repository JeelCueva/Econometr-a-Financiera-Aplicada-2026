# Capítulo 15, Ejercicio 15.2: la hipótesis de las expectativas (regresiones de Campbell y Shiller)
import numpy as np
from cap15_funciones import cargar, mco_nw, TAU

f, Y = cargar(); T = len(Y); col = {m: j for j, m in enumerate(TAU)}
print("(a) Tasas cortas futuras: (1/k) sum_{i=0}^{k-1} y_{m,t+m i} - y_{m,t} = a + b (y_{k,t} - y_{m,t})")
print("    H0 (expectativas): b = 1; errores de Newey-West con k rezagos")
print("   m     k     n       b      ee(b)   t(b=1)     R2")
res = []
for m, k in [(1, 3), (1, 6), (1, 12), (12, 36), (12, 60), (12, 120)]:
    ym, yk = Y[:, col[m]], Y[:, col[k]]; h = k // m; n = T - (k - m)
    izq = np.array([ym[t + m * np.arange(h)].mean() for t in range(n)]) - ym[:n]
    b, ee, r2, _ = mco_nw(izq, yk[:n] - ym[:n], k); res.append((m, k, b[1], ee[1]))
    print(f"{m:4d} {k:5d} {n:5d} {b[1]:8.4f} {ee[1]:8.4f} {(b[1] - 1) / ee[1]:8.2f} {r2:8.3f}")
print("\n(b) Tasa larga: y_{11,t+1} - y_{12,t} = a + b (y_{12,t} - y_{1,t}) / 11;  H0: b = 1")
y12, y11, y1 = Y[:, col[12]], Y[:, col[11]], Y[:, col[1]]
b, ee, r2, n = mco_nw(y11[1:] - y12[:-1], (y12[:-1] - y1[:-1]) / 11, 1)
print(f"    b = {b[1]:.4f} (ee {ee[1]:.4f}); t(b = 1) = {(b[1] - 1) / ee[1]:.2f}; t(b = 0) = {b[1] / ee[1]:.2f}; R2 = {r2:.4f}; n = {n}")

# ---- Figura: coeficientes b con intervalos al 95 % ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 3.4)); x = np.arange(len(res))
ax.errorbar(x, [r[2] for r in res], yerr=[1.96 * r[3] for r in res], fmt="o", color=eg.AZUL, capsize=3)
ax.axhline(1, color=eg.LADRILLO, ls="--", lw=1, label="hipótesis de las expectativas ($b=1$)"); ax.axhline(0, color="black", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([f"{r[0]}→{r[1]}" for r in res]); ax.set_xlabel("$m$→$k$ (meses)"); ax.set_ylabel("$\\hat b$")
ax.legend(fontsize=8); eg.coma(ax); ax.set_xticklabels([f"{r[0]}→{r[1]}" for r in res]); fig.savefig("graficos/cap15_ej02_expectativas.pdf")
