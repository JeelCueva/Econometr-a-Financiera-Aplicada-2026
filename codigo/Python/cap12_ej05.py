# Capítulo 12, Ejercicio 12.5: ¿contagio en 2008? La corrección de Forbes y Rigobon
import numpy as np
from cap12_funciones import panel_diario

f, R = panel_diario()
calma = (f >= "2006-01-01") & (f <= "2008-08-31"); crisis = (f >= "2008-09-01") & (f <= "2009-03-31")
nt, nc = calma.sum(), crisis.sum(); x = R[:, 0]
delta = x[crisis].var(ddof=1) / x[calma].var(ddof=1) - 1
print(f"Calma: {nt} días; crisis: {nc} días. Varianza diaria del S&P 500: {x[calma].var(ddof=1):.4f} en calma y "
      f"{x[crisis].var(ddof=1):.4f} en crisis; delta = {delta:.4f}\n")
print("Mercado       rho calma   rho crisis   rho crisis ajustada   z (sin ajuste)   z (ajustada)")
for j, nm in [(1, "Nasdaq"), (2, "Brent")]:
    y = R[:, j]; rt = np.corrcoef(x[calma], y[calma])[0, 1]; rc = np.corrcoef(x[crisis], y[crisis])[0, 1]
    ra = rc / np.sqrt(1 + delta * (1 - rc**2)); ee = np.sqrt(1 / (nc - 3) + 1 / (nt - 3))
    print(f"{nm:<10} {rt:11.4f} {rc:12.4f} {ra:20.4f} {(np.arctanh(rc) - np.arctanh(rt)) / ee:16.2f} {(np.arctanh(ra) - np.arctanh(rt)) / ee:14.2f}")

# ---- Figura: S&P 500 frente al Brent en calma y en crisis ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True, sharey=True)
for a, k, tit, col in [(ax[0], calma, "calma: 2006-01 a 2008-08", eg.AZUL), (ax[1], crisis, "crisis: 2008-09 a 2009-03", eg.LADRILLO)]:
    a.plot(x[k], R[k, 2], ".", color=col, ms=3, alpha=0.7)
    b = np.polyfit(x[k], R[k, 2], 1); xs = np.linspace(-10, 11, 20); a.plot(xs, np.polyval(b, xs), color="black", lw=0.9)
    a.set_title(f"{tit}; $r={np.corrcoef(x[k], R[k, 2])[0, 1]:.2f}$".replace(".", "{,}"), fontsize=9)
    a.set_xlabel("rendimiento del S&P 500 (%)"); eg.coma(a)
ax[0].set_ylabel("rendimiento del Brent (%)")
fig.tight_layout(); fig.savefig("graficos/cap12_ej05_contagio.pdf")
