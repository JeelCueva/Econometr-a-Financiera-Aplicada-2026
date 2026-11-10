# Capítulo 21, Etapa 1: los datos y sus hechos estilizados
import numpy as np
from scipy import stats
from cap21_funciones import cargar, NOMBRES, asimetria, curtosis, ljung_box

f, R = cargar(); n = len(f)
print(f"Rendimientos simples diarios (%), {f[0]} a {f[-1]}; n = {n}\n")
print("serie    media anual   vol. anual   asimetría   curtosis    JB        LB(10) r   p      LB(10) r2   p      peor día")
for j, nom in enumerate(NOMBRES):
    x = R[:, j]; s, k = asimetria(x), curtosis(x); jb = n / 6 * (s**2 + (k - 3)**2 / 4)
    q1, p1 = ljung_box(x, 10); q2, p2 = ljung_box(x**2, 10); i = np.argmin(x)
    print(f"{nom:6s}  {252 * x.mean():10.2f}  {np.sqrt(252) * x.std(ddof=1):10.2f}  {s:10.3f}  {k:9.2f}  {jb:9.1f}  {q1:9.2f}  {p1:6.4f}  {q2:9.2f}  {p2:6.4f}   {x[i]:7.2f} ({f[i]})")
print("\nMatriz de correlaciones:")
C = np.corrcoef(R.T); print("        " + " ".join(f"{v:>7s}" for v in NOMBRES))
for j, nom in enumerate(NOMBRES): print(f"{nom:6s}  " + " ".join(f"{c:7.3f}" for c in C[j]))
print(f"\nValor crítico de JB al 1 % (chi2 con 2 gl) = {stats.chi2.ppf(0.99, 2):.2f}; de LB(10) = {stats.chi2.ppf(0.99, 10):.2f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8.6, 3.4)); t = np.arange(n); cols = [eg.AZUL, eg.LADRILLO, eg.OLIVA, eg.GRIS, "black"]
for j, nom in enumerate(NOMBRES): ax.plot(t, 100 * np.cumprod(1 + R[:, j] / 100), color=cols[j], lw=1.6 if nom == "SP500" else 1.0, ls="--" if nom == "SP500" else "-", label=nom)
pos = [k for k in t if (k == 0 or f[k][:4] != f[k - 1][:4])]; ax.set_xticks(pos); ax.set_ylabel("valor de 100 invertidos"); ax.legend(fontsize=8, ncol=5)
eg.coma(ax); ax.set_xticklabels([f[k][:4] for k in pos]); fig.savefig("graficos/cap21_ej01_riqueza.pdf")
