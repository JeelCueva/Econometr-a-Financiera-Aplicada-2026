# Capítulo 19, Ejercicio 19.5: efectos fijos de cartera y de mes: momentum de series de tiempo frente a momentum de corte transversal
import numpy as np
from cap19_funciones import panel_momentum, mco, media_grupo, v_cluster, v_dos_vias

y, x, c, m, f = panel_momentum(); n = len(y); N = len(np.unique(c)); T = len(np.unique(m))
def quitar(z, cartera, mes):
    if cartera and mes: return z - media_grupo(z, c) - media_grupo(z, m) + z.mean()
    if cartera: return z - media_grupo(z, c)
    if mes: return z - media_grupo(z, m)
    return z - z.mean()
print("especificación                    pendiente   ee(mes)   ee(dos vías)   t(dos vías)   varianza de x que usa")
res = {}
for nom, cc, mm in [("agrupado (sin efectos)", False, False), ("efectos de cartera", True, False), ("efectos de mes", False, True), ("cartera y mes", True, True)]:
    yt, xt = quitar(y, cc, mm), quitar(x, cc, mm); X = xt[:, None]; b, e, A = mco(yt, X); k = 1 + (N - 1) * cc + (T - 1) * mm + 1
    sm, sd = np.sqrt(v_cluster(X, e, A, m, k)[0, 0]), np.sqrt(v_dos_vias(X, e, A, c, m, k)[0, 0]); res[nom] = b[0]
    print(f"  {nom:30s}  {b[0]:9.5f}  {sm:8.5f}  {sd:10.5f}  {b[0] / sd:10.2f}      {(xt @ xt) / ((x - x.mean()) @ (x - x.mean())):.4f}")
xm = media_grupo(x, m); xc = media_grupo(x, c); ym = media_grupo(y, m)
_, idx = np.unique(m, return_index=True); bt, _, _ = mco(ym[idx], np.column_stack([np.ones(T), xm[idx]]))
print(f"\nRegresión de series de tiempo del promedio de las 30 carteras sobre su propio x promedio: pendiente = {bt[1]:.5f}")
w = [((quitar(x, False, True))**2).sum(), ((xm - x.mean())**2).sum()]
print(f"Descomposición de la varianza de x: dentro de cada mes = {w[0] / sum(w):.4f}; entre meses = {w[1] / sum(w):.4f}")
print(f"Pendiente agrupada = {w[0] / sum(w):.4f} x {res['efectos de mes']:.5f} + {w[1] / sum(w):.4f} x {bt[1]:.5f} = {w[0] / sum(w) * res['efectos de mes'] + w[1] / sum(w) * bt[1]:.5f}")
print(f"\nInterpretación económica de la pendiente con efectos de mes: 10 puntos más de rendimiento pasado -> {10 * res['efectos de mes']:.3f} puntos más de exceso de rendimiento mensual")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.5))
xd, yd = quitar(x, False, True), quitar(y, False, True); ax[0].plot(xd, yd, ".", ms=1, color=eg.AZUL, alpha=0.25)
g_ = np.linspace(-60, 60, 10); ax[0].plot(g_, res["efectos de mes"] * g_, color=eg.LADRILLO, lw=1.5); ax[0].set_xlim(-60, 60); ax[0].set_ylim(-20, 20)
ax[0].set_title("Dentro de cada mes (corte transversal)", fontsize=9); ax[0].set_xlabel("$x$ menos su media del mes"); ax[0].set_ylabel("$y$ menos su media del mes")
ax[1].plot(xm[idx], ym[idx], ".", ms=2.5, color=eg.AZUL, alpha=0.5); g_ = np.linspace(xm.min(), xm.max(), 10); ax[1].plot(g_, bt[0] + bt[1] * g_, color=eg.LADRILLO, lw=1.5)
ax[1].set_title("Entre meses (promedios del mes)", fontsize=9); ax[1].set_xlabel("$x$ promedio del mes"); ax[1].set_ylabel("$y$ promedio del mes")
for a in ax: eg.coma(a)
fig.tight_layout(); fig.savefig("graficos/cap19_ej05_descomposicion.pdf")
