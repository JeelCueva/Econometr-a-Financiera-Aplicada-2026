# Capítulo 20, Ejercicio 20.1: predicción de la prima de mercado con un predictor a la vez (Goyal y Welch, 2008)
import numpy as np
from cap20_funciones import cargar, VARS, r2_os, clark_west

mes, X, y = cargar(); n = len(y); i0 = int(np.where(mes == "1957-01")[0][0])
print(f"Objetivo: exceso de rendimiento del mercado del mes siguiente (%), {mes[0]} a {mes[-1]}; n = {n}")
print(f"Fuera de muestra: {mes[i0]} a {mes[-1]} ({n - i0} meses), ventana creciente, reestimación mensual\n")
fb = np.array([y[:i].mean() for i in range(i0, n)]); yo = y[i0:]
print("predictor   pendiente   t(White)   R2 dentro (%)   R2 fuera (%)   R2 fuera CT (%)   CW      p(CW)")
F = {}
for j, v in enumerate(VARS):
    x = X[:, j]; Z = np.column_stack([np.ones(n), x]); A = np.linalg.inv(Z.T @ Z); b = A @ Z.T @ y; e = y - Z @ b
    V = n / (n - 2) * A @ (Z.T * e**2) @ Z @ A; r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    f = np.empty(n - i0)
    for k, i in enumerate(range(i0, n)):
        Zi = np.column_stack([np.ones(i), x[:i]])           # todos los pares anteriores al mes objetivo
        bi = np.linalg.solve(Zi.T @ Zi, Zi.T @ y[:i]); f[k] = bi[0] + bi[1] * x[i]
    F[v] = f; fct = np.maximum(f, 0); cw, p = clark_west(yo, f, fb)
    print(f"  {v:7s}  {b[1]:10.4f}  {b[1] / np.sqrt(V[1, 1]):8.2f}  {100 * r2:12.2f}  {100 * r2_os(yo, f, fb):13.2f}  {100 * r2_os(yo, fct, fb):15.2f}  {cw:6.2f}  {p:7.4f}")
np.save("salidas/cap20_pronosticos_univariados.npy", np.column_stack([F[v] for v in VARS]))

# ---- Figura: diferencia acumulada de errores cuadráticos (Goyal y Welch) ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8.6, 3.4)); t = np.arange(n - i0)
for v, c, ls in [("dp", eg.AZUL, "-"), ("cape", eg.LADRILLO, "--"), ("tms", eg.OLIVA, "-."), ("infl", eg.GRIS, ":")]:
    ax.plot(t, np.cumsum((yo - fb)**2 - (yo - F[v])**2), color=c, ls=ls, lw=1.2, label=v)
ax.axhline(0, color="black", lw=0.5); pos = [k for k in t if mes[i0 + k].endswith("-01") and int(mes[i0 + k][:4]) % 10 == 0]
ax.set_xticks(pos); ax.set_xticklabels([mes[i0 + k][:4] for k in pos]); ax.set_ylabel("SCE(media) - SCE(modelo)"); ax.legend(fontsize=8, ncol=4)
eg.coma(ax); ax.set_xticklabels([mes[i0 + k][:4] for k in pos]); fig.savefig("graficos/cap20_ej01_goyal_welch.pdf")
