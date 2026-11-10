# Capítulo 16, Ejercicio 16.4: tamaño y potencia de las pruebas (simulación de Brown y Warner con datos reales)
import numpy as np
from cap16_funciones import cargar, rendimientos_normales, pruebas, corrado, LCG

f, R = cargar(); m = R[:, 4]; T = len(m); F0 = np.arange(250, T)      # días 0 posibles
E, AR0, V0 = {}, {}, {}
for j in range(4):
    for t0 in F0:
        x = rendimientos_normales(R[:, j], m, t0, evt=(0, 0)); E[j, t0] = x["e"]; AR0[j, t0] = x["AR"][0]; V0[j, t0] = x["V"][0, 0]
GL = 238; N, REP = 20, 1000; DELTAS = [0.0, 0.5, 1.0, 2.0]
g = LCG(20161); sorteo = [[(int(4 * g.u()), int(F0[0] + int(len(F0) * g.u()))) for _ in range(N)] for _ in range(REP)]
print(f"{len(F0)} fechas posibles x 4 acciones; {REP} muestras de N = {N} eventos; rechazo si |estadístico| > 1,96\n")
print("delta (%)    J1     Patell    BMP    signo   Corrado")
tasas = {}
for d in DELTAS:
    rech = np.zeros(5)
    for s in sorteo:
        a = np.array([AR0[k] for k in s]) + d; v = np.array([V0[k] for k in s])
        p = pruebas(a, v, np.full(N, GL)); A = np.array([np.r_[E[k], a[i]] for i, k in enumerate(s)])
        est = np.array([p["J1"], p["Patell"], p["BMP"], p["signo"], corrado(A)]); rech += np.abs(est) > 1.96
    tasas[d] = rech / REP
    print(f"{d:6.1f}   " + "  ".join(f"{x:7.3f}" for x in tasas[d]))
a = np.array([AR0[j, t] for j in range(4) for t in F0])
print(f"\nAR(0) de todas las fechas posibles: media = {a.mean():.4f}; mediana = {np.median(a):.4f}; fracción positiva = {(a > 0).mean():.4f}; "
      f"asimetría = {((a - a.mean())**3).mean() / a.std()**3:.4f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for k, (nom, c, ls) in enumerate([("$J_1$", eg.AZUL, "-"), ("Patell", eg.GRIS, "--"), ("BMP", eg.LADRILLO, "-."), ("signo", eg.OLIVA, ":"), ("Corrado", "black", "-")]):
    ax.plot(DELTAS, [tasas[d][k] for d in DELTAS], marker="o", ms=3, color=c, ls=ls, lw=1.3, label=nom)
ax.axhline(0.05, color=eg.GRIS, lw=0.8, ls=":"); ax.set_xlabel("rendimiento anormal agregado $\\delta$ (%)"); ax.set_ylabel("frecuencia de rechazo")
ax.legend(fontsize=8); eg.coma(ax); fig.savefig("graficos/cap16_ej04_potencia.pdf")
