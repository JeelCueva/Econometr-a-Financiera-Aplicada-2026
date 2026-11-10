# Capítulo 17, Ejercicio 17.5: pronóstico de un mes y VaR con el modelo de dos regímenes
import numpy as np
from scipy.stats import norm, chi2
from cap17_funciones import mercado, em_media_varianza, cuantil_mezcla, kupiec_christoffersen

f, y = mercado(); T = len(y); m, sd = y.mean(), y.std(ddof=0)
r = em_media_varianza(y, 2, [m, m], [0.7 * sd, 1.5 * sd], [[0.95, 0.05], [0.05, 0.95]])
mu, s, P = r["mu"], r["s"], r["P"]
w = r["xf"][-1] @ P; mp = w @ mu; vp = w @ (s**2 + mu**2) - mp**2
print(f"Pronóstico para el mes siguiente a {f[-1]}:")
print(f"  Pr filtrada del régimen volátil en {f[-1]} = {r['xf'][-1, 1]:.4f}; Pr predicha = {w[1]:.4f}")
print(f"  media = {mp:.4f} %; desv. est. = {np.sqrt(vp):.4f} %")
for a in (0.05, 0.01):
    print(f"  VaR al {100 * (1 - a):.0f} %: mezcla = {-cuantil_mezcla(a, w, mu, s):.4f} %; normal incondicional = {-(m + sd * norm.ppf(a)):.4f} %")
# Prueba retrospectiva dentro de la muestra (parámetros de toda la muestra; probabilidades solo con información hasta t)
W = r["xf"][:-1] @ P; y1 = y[1:]
print(f"\nPrueba retrospectiva de {len(y1)} meses ({f[1]} a {f[-1]}); parámetros estimados con toda la muestra")
print("  nivel  modelo        excepciones  esperadas   frecuencia   LR_uc   p_uc    LR_ind   p_ind")
for a in (0.05, 0.01):
    qm = np.array([cuantil_mezcla(a, W[t], mu, s) for t in range(len(y1))]); qn = m + sd * norm.ppf(a)
    for nom, q in [("dos regímenes", qm), ("normal      ", np.full(len(y1), qn))]:
        h = (y1 < q).astype(int); uc, ind, _ = kupiec_christoffersen(h, a)
        print(f"  {a:5.2f}  {nom}  {h.sum():8d}  {a * len(y1):10.1f}  {h.mean():10.4f}  {uc:7.2f} {chi2.sf(uc, 1):6.4f}  {ind:7.2f} {chi2.sf(ind, 1):6.4f}")
    if a == 0.01: Q1 = qm
print(f"\nVaR al 99 % de la mezcla: mínimo = {-Q1.max():.4f} %; máximo = {-Q1.min():.4f} %; mediana = {-np.median(Q1):.4f} %")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
t = np.arange(1, T); pos = [k for k in range(T) if f[k].endswith("-01") and int(f[k][:4]) % 10 == 0]
fig, ax = plt.subplots(figsize=(8.6, 3.2))
ax.plot(t, y1, color=eg.GRIS, lw=0.5, label="exceso de rendimiento")
ax.plot(t, Q1, color=eg.LADRILLO, lw=0.9, label="VaR al 99 %, dos regímenes (con signo)")
ax.axhline(m + sd * norm.ppf(0.01), color=eg.AZUL, ls="--", lw=0.9, label="VaR al 99 %, normal")
h = y1 < Q1; ax.plot(t[h], y1[h], "o", ms=3, color=eg.LADRILLO, mfc="none")
ax.set_xticks(pos); ax.set_xticklabels([f[k][:4] for k in pos]); ax.set_ylabel("%"); ax.legend(fontsize=7, loc="lower right")
eg.coma(ax); ax.set_xticklabels([f[k][:4] for k in pos]); fig.savefig("graficos/cap17_ej05_var.pdf")
