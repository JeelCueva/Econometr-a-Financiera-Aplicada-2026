# Capítulo 17, Ejercicio 17.4: ¿cuántos regímenes? Comparación de modelos para el mercado de EE. UU.
import numpy as np
from scipy.stats import chi2
from cap17_funciones import mercado, em_media_varianza, ergodicas

f, y = mercado(); T = len(y); m, sd = y.mean(), y.std(ddof=0)
lln = -0.5 * T * (np.log(2 * np.pi * sd**2) + 1)
P2 = [[0.95, 0.05], [0.05, 0.95]]; P3 = [[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]]
M = {"1 régimen (normal)": (lln, 2, None),
     "2 regímenes, solo varianza": None, "2 regímenes, media y varianza": None, "3 regímenes, media y varianza": None}
r2v = em_media_varianza(y, 2, [m, m], [0.7 * sd, 1.5 * sd], P2, media_comun=True)
r2 = em_media_varianza(y, 2, [m, m], [0.7 * sd, 1.5 * sd], P2)
r3 = em_media_varianza(y, 3, [m, m, m], [0.5 * sd, 1.0 * sd, 2.0 * sd], P3)
M["2 regímenes, solo varianza"] = (r2v["ll"], 5, r2v); M["2 regímenes, media y varianza"] = (r2["ll"], 6, r2); M["3 regímenes, media y varianza"] = (r3["ll"], 12, r3)
print(f"T = {T}\n  k    log-verosim.       AIC        BIC   modelo")
for nom, (ll, k, _) in M.items():
    print(f"{k:3d} {ll:14.4f} {-2 * ll + 2 * k:10.2f} {-2 * ll + k * np.log(T):10.2f}   {nom}")
lr = 2 * (r2["ll"] - r2v["ll"])
print(f"\nLR de medias iguales entre regímenes (2 regímenes): {lr:.4f}; valor p chi2(1) = {chi2.sf(lr, 1):.4f}")
print(f"  modelo de solo varianza: mu = {r2v['mu'][0]:.4f}; sigma = {r2v['s'][0]:.4f} {r2v['s'][1]:.4f}; p11 = {r2v['P'][0, 0]:.4f}; p22 = {r2v['P'][1, 1]:.4f}")
print("\nModelo de 3 regímenes:")
for j in range(3): print(f"  régimen {j + 1}: mu = {r3['mu'][j]:8.4f}; sigma = {r3['s'][j]:8.4f}; p_jj = {r3['P'][j, j]:.4f}; duración = {1 / (1 - r3['P'][j, j]):7.2f} meses")
print("  matriz de transición:"); [print("   " + " ".join(f"{v:8.4f}" for v in fila)) for fila in r3["P"]]
pi3 = ergodicas(r3["P"]); print("  probabilidades estacionarias: " + " ".join(f"{v:.4f}" for v in pi3))
c3 = r3["xs"].argmax(1); print("  meses asignados (máxima Pr suavizada): " + " ".join(str((c3 == j).sum()) for j in range(3)))
for j in (2,):
    k = np.where(c3 == j)[0]; print(f"  primeros y últimos meses del régimen {j + 1}: {f[k[0]]} ... {f[k[-1]]}; décadas: " + " ".join(sorted(set(x[:3] + '0' for x in f[k]))))

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
t = np.arange(T); pos = [k for k in range(T) if f[k].endswith("-01") and int(f[k][:4]) % 10 == 0]
fig, ax = plt.subplots(figsize=(8.6, 2.9))
ax.stackplot(t, r3["xs"].T, colors=[eg.AZUL, eg.GRIS, eg.LADRILLO], alpha=0.8, labels=[f"régimen {j + 1} ($\\sigma$ = {r3['s'][j]:.1f})".replace(".", ",") for j in range(3)])
ax.set_xticks(pos); ax.set_xticklabels([f[k][:4] for k in pos]); ax.set_ylabel("Pr suavizada"); ax.set_ylim(0, 1); ax.set_xlim(0, T - 1)
ax.legend(fontsize=7, loc="lower left", ncol=3); eg.coma(ax); ax.set_xticklabels([f[k][:4] for k in pos]); fig.savefig("graficos/cap17_ej04_tres_regimenes.pdf")
