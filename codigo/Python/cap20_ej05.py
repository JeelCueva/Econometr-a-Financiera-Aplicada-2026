# Capítulo 20, Ejercicio 20.5: árboles de regresión y sobreajuste
import numpy as np
from cap20_funciones import cargar, VARS_MULT, IDX_MULT, arbol, predecir, imprimir_arbol

mes, X, y = cargar(); X = X[:, IDX_MULT]; corte = int(np.where(mes == "1981-01")[0][0])
Xa, ya, Xp, yp = X[:corte], y[:corte], X[corte:], y[corte:]
print(f"Entrenamiento: {mes[0]} a {mes[corte - 1]} ({corte} meses); prueba: {mes[corte]} a {mes[-1]} ({len(yp)} meses); hoja mínima = 24 meses")
ref = ((yp - ya.mean())**2).mean()
Z = np.column_stack([np.ones(corte), Xa]); b = np.linalg.solve(Z.T @ Z, Z.T @ ya)
ecm_l = ((yp - np.column_stack([np.ones(len(yp)), Xp]) @ b)**2).mean(); ecm_la = ((ya - Z @ b)**2).mean()
print(f"\nMCO lineal con 8 predictores: ECM entrenamiento = {ecm_la:.4f}; ECM prueba = {ecm_l:.4f}; R2 prueba = {100 * (1 - ecm_l / ref):.2f} %")
print("\nprofundidad   hojas   ECM entrenamiento   ECM prueba   R2 prueba (%)")
res = []
for d in range(0, 7):
    t = arbol(Xa, ya, d); ea = ((ya - predecir(t, Xa))**2).mean(); ep = ((yp - predecir(t, Xp))**2).mean()
    hojas = len(np.unique(predecir(t, Xa))); res.append((d, ea, ep)); print(f"   {d:5d}      {hojas:4d}     {ea:12.4f}     {ep:10.4f}    {100 * (1 - ep / ref):8.2f}")
print("\nÁrbol de profundidad 2:"); imprimir_arbol(arbol(Xa, ya, 2), VARS_MULT)

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
r = np.array(res); fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.plot(r[:, 0], r[:, 1], "o-", color=eg.AZUL, lw=1.4, ms=4, label="entrenamiento (1927-1980)"); ax.plot(r[:, 0], r[:, 2], "s--", color=eg.LADRILLO, lw=1.4, ms=4, label="prueba (1981-2018)")
ax.axhline(ref, color=eg.GRIS, ls=":", lw=1, label="prueba con la media de entrenamiento"); ax.set_xlabel("profundidad del árbol"); ax.set_ylabel("error cuadrático medio")
ax.legend(fontsize=8); eg.coma(ax); fig.savefig("graficos/cap20_ej05_arboles.pdf")
