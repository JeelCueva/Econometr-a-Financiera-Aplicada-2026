# Capítulo 19, Ejercicio 19.4: ¿qué errores estándar? Petersen (2009) con un panel de 30 carteras de French
import numpy as np
from cap19_funciones import panel_momentum, mco, v_clasica, v_white, v_cluster, v_dos_vias, v_driscoll_kraay

y, x, c, m, f = panel_momentum(); n = len(y); N = len(np.unique(c)); T = len(np.unique(m))
print(f"Panel: N = {N} carteras, T = {T} meses ({f[m.min()]} a {f[m.max()]}), n = {n}")
print(f"y = exceso de rendimiento del mes t (%); x = rendimiento acumulado de t-12 a t-2 (%): media = {x.mean():.4f}; desv. est. = {x.std(ddof=1):.4f}")
X = np.column_stack([np.ones(n), x]); b, e, A = mco(y, X)
E = e.reshape(T, N); Xs = x.reshape(T, N)
def corr_media(M, eje):
    if eje == "tiempo":    # misma cartera, meses distintos: autocorrelación de orden 1
        return np.mean([np.corrcoef(M[1:, j], M[:-1, j])[0, 1] for j in range(N)])
    C = np.corrcoef(M.T); return C[np.triu_indices(N, 1)].mean()  # mismo mes, carteras distintas
print(f"\nCorrelación media de los residuos entre carteras en el mismo mes = {corr_media(E, 'corte'):.4f}; de x = {corr_media(Xs, 'corte'):.4f}")
print(f"Autocorrelación media de orden 1 de los residuos de cada cartera = {corr_media(E, 'tiempo'):.4f}; de x = {corr_media(Xs, 'tiempo'):.4f}")
print(f"\nMCO agrupado: constante = {b[0]:.4f}; pendiente = {b[1]:.5f}")
V = {"clásicos": v_clasica(X, e, A, n - 2), "White (HC1)": v_white(X, e, A, 2), "agrupados por cartera": v_cluster(X, e, A, c, 2),
     "agrupados por mes": v_cluster(X, e, A, m, 2), "dos vías": v_dos_vias(X, e, A, c, m, 2), "Driscoll-Kraay (L = 12)": v_driscoll_kraay(X, e, A, m, 12)}
print("   ee(pendiente)   t      cociente con el clásico   método")
s0 = np.sqrt(V["clásicos"][1, 1])
for nom, Vm in V.items():
    s = np.sqrt(Vm[1, 1]); print(f"   {s:.5f}   {b[1] / s:7.2f}   {s / s0:10.3f}                  {nom}")
# Fama-MacBeth
B = np.array([np.polyfit(Xs[k], (y.reshape(T, N))[k], 1)[0] for k in range(T)])
fm, sfm = B.mean(), B.std(ddof=1) / np.sqrt(T)
h = B - fm; S = (h @ h) / T
for j in range(1, 13): S += 2 * (1 - j / 13) * (h[j:] @ h[:-j]) / T
snw = np.sqrt(S / T)
print(f"   {sfm:.5f}   {fm / sfm:7.2f}   {sfm / s0:10.3f}                  Fama-MacBeth (pendiente media = {fm:.5f})")
print(f"   {snw:.5f}   {fm / snw:7.2f}   {snw / s0:10.3f}                  Fama-MacBeth con Newey-West (12 rezagos)")
print(f"\nautocorrelación de orden 1 de las pendientes mensuales de Fama-MacBeth = {np.corrcoef(B[1:], B[:-1])[0, 1]:.4f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
nom = list(V.keys()) + ["Fama-MacBeth", "FM con Newey-West"]; val = [np.sqrt(V[k][1, 1]) for k in V] + [sfm, snw]
fig, ax = plt.subplots(figsize=(6.6, 3.4)); yy = np.arange(len(nom))
ax.barh(yy, val, color=[eg.GRIS, eg.GRIS, eg.AZUL, eg.LADRILLO, eg.LADRILLO, eg.LADRILLO, eg.OLIVA, eg.OLIVA])
ax.invert_yaxis(); ax.set_xlabel("error estándar de la pendiente"); eg.coma(ax); ax.set_yticks(yy); ax.set_yticklabels(nom, fontsize=8)
fig.tight_layout(); fig.savefig("graficos/cap19_ej04_errores.pdf")
