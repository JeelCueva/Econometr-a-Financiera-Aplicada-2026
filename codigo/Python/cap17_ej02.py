# Capítulo 17, Ejercicio 17.2: el régimen volátil y las recesiones del NBER
import numpy as np
from cap17_funciones import mercado, em_media_varianza, recesion_mensual

f, y = mercado(); T = len(y); m, sd = y.mean(), y.std(ddof=0)
r = em_media_varianza(y, 2, [m, m], [0.7 * sd, 1.5 * sd], [[0.95, 0.05], [0.05, 0.95]])
p2 = r["xs"][:, 1]; rec = recesion_mensual(f); c = (p2 > 0.5).astype(int)
print(f"meses en recesión (NBER): {rec.sum()} de {T}")
print(f"Pr media del régimen volátil: en recesión = {p2[rec == 1].mean():.4f}; en expansión = {p2[rec == 0].mean():.4f}")
print("\nTabla de clasificación (filas: NBER; columnas: régimen con Pr > 0,5)")
print("              tranquilo   volátil")
for k, nom in [(0, "expansión"), (1, "recesión ")]: print(f"  {nom}   {((rec == k) & (c == 0)).sum():8d}  {((rec == k) & (c == 1)).sum():8d}")
print(f"Pr(volátil | recesión) = {c[rec == 1].mean():.4f}; Pr(volátil | expansión) = {c[rec == 0].mean():.4f}")
print(f"correlación entre Pr suavizada e indicador de recesión = {np.corrcoef(p2, rec)[0, 1]:.4f}")
print("\nEpisodios del régimen volátil (Pr suavizada > 0,5) de 6 meses o más:")
print("  inicio    fin      meses  rend. medio  desv. est.  meses en recesión")
i = 0
while i < T:
    if c[i]:
        j = i
        while j + 1 < T and c[j + 1]: j += 1
        if j - i + 1 >= 6:
            seg = y[i:j + 1]; print(f"  {f[i]}  {f[j]}  {j - i + 1:5d}  {seg.mean():10.4f}  {seg.std(ddof=0):10.4f}  {rec[i:j + 1].sum():8d}")
        i = j + 1
    else: i += 1
print("\nPr media del régimen volátil en cada recesión desde 1929:")
import pandas as pd
R = pd.read_csv("datos/internacional/recesiones_nber.csv")
for p, v in zip(R["pico"], R["valle"]):
    k = (f > p) & (f <= v)
    if k.sum() and p >= "1929": print(f"  {p} a {v}: {p2[k].mean():.4f}  ({k.sum()} meses)")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
t = np.arange(T); pos = [k for k in range(T) if f[k].endswith("-01") and int(f[k][:4]) % 10 == 0]
fig, ax = plt.subplots(figsize=(8.6, 3.0))
for k in range(T):
    if rec[k]: ax.axvspan(k - 0.5, k + 0.5, color=eg.GRIS, alpha=0.35, lw=0)
ax.plot(t, p2, color=eg.LADRILLO, lw=0.8, label="Pr(régimen volátil), suavizada")
ax.set_xticks(pos); ax.set_xticklabels([f[k][:4] for k in pos]); ax.set_ylabel("probabilidad"); ax.legend(fontsize=8, loc="upper right")
eg.coma(ax); ax.set_xticklabels([f[k][:4] for k in pos]); fig.savefig("graficos/cap17_ej02_nber.pdf")
