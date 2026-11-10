# Capítulo 16, Ejercicio 16.2: el caso Cambridge Analytica (19 de marzo de 2018) y el contagio a otras tecnológicas
import numpy as np
from scipy.stats import norm
from cap16_funciones import cargar, rendimientos_normales, car, ACT

f, R = cargar(); m = R[:, 4]; t0 = int(np.where(f == "2018-03-19")[0][0])
print(f"día 0 = {f[t0]}; ventana del evento: {f[t0 - 5]} a {f[t0 + 5]}; modelo de mercado, estimación [-250, -11]\n")
print("activo    beta     s      AR(0)    t     CAR(0,+1)    t     CAR(0,+5)    t")
res = {}
for j, a in enumerate(ACT):
    r = rendimientos_normales(R[:, j], m, t0); res[a] = r; c1, c5 = car(r, 0, 1), car(r, 0, 5)
    print(f"{a:6s} {r['b'][1]:7.4f} {np.sqrt(r['s2']):6.4f} {r['AR'][5]:8.4f} {r['AR'][5] / np.sqrt(r['V'][5, 5]):6.2f} "
          f"{c1[0]:9.4f} {c1[0] / c1[1]:6.2f} {c5[0]:10.4f} {c5[0] / c5[1]:6.2f}")
pares = ["AAPL", "AMZN", "GOOG"]; port = R[:, [0, 1, 3]].mean(1)
rp = rendimientos_normales(port, m, t0); c5 = car(rp, 0, 5)
print(f"\nCartera igualmente ponderada de AAPL, AMZN y GOOG: beta = {rp['b'][1]:.4f}; s = {np.sqrt(rp['s2']):.4f}")
print(f"  CAR(0,+5) = {c5[0]:.4f}; ee = {c5[1]:.4f}; t = {c5[0] / c5[1]:.2f}")
E = np.column_stack([res[a]["e"] for a in pares]); C = np.corrcoef(E.T)
print(f"Correlaciones de los residuos en la estimación: AAPL-AMZN {C[0, 1]:.4f}, AAPL-GOOG {C[0, 2]:.4f}, AMZN-GOOG {C[1, 2]:.4f}")
cars = np.array([car(res[a], 0, 5)[0] for a in pares]); vs = np.array([car(res[a], 0, 5)[1]**2 for a in pares])
ing = cars.mean() / np.sqrt(vs.sum() / 9)
print(f"Prueba ingenua (supone independencia): CAAR(0,+5) = {cars.mean():.4f}; ee = {np.sqrt(vs.sum() / 9):.4f}; t = {ing:.2f}")
print(f"Cociente de errores estándar cartera/ingenuo = {c5[1] / np.sqrt(vs.sum() / 9):.4f}")
dif = R[:, 2] - port; rd = rendimientos_normales(dif, m, t0, modelo="media"); cd = car(rd, 0, 5)
print(f"\nFB menos la cartera de pares (modelo de media): CAR(0,+5) = {cd[0]:.4f}; ee = {cd[1]:.4f}; t = {cd[0] / cd[1]:.2f}; "
      f"valor p = {2 * norm.sf(abs(cd[0] / cd[1])):.2e}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.6, 3.4)); tau = res["FB"]["tau"]
for a, c, ls in [("FB", eg.LADRILLO, "-"), ("AAPL", eg.AZUL, "--"), ("AMZN", eg.OLIVA, ":"), ("GOOG", eg.GRIS, "-.")]:
    ax.plot(tau, np.cumsum(res[a]["AR"]), color=c, ls=ls, lw=1.5, marker="o", ms=3, label=a)
ax.axhline(0, color="black", lw=0.5); ax.axvline(-0.5, color="black", lw=0.5, ls=":")
ax.set_xlabel("$\\tau$ (días hábiles; 0 = 19 de marzo de 2018)"); ax.set_ylabel("CAR$(-5,\\tau)$ (%)"); ax.legend(fontsize=8); eg.coma(ax)
fig.savefig("graficos/cap16_ej02_cambridge.pdf")
