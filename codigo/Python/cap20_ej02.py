# Capítulo 20, Ejercicio 20.2: todos los predictores a la vez: MCO, ridge, lasso y red elástica fuera de muestra
import numpy as np
from cap20_funciones import cargar, VARS_MULT, IDX_MULT, estandarizar, ajustar_validado, r2_os, clark_west

mes, X, y = cargar(); X = X[:, IDX_MULT]; n = len(y); i0 = int(np.where(mes == "1957-01")[0][0])
fb = np.array([y[:i].mean() for i in range(i0, n)]); yo = y[i0:]
anios = list(range(i0, n, 12)); M = {"MCO (todos)": None, "ridge": 0.0, "red elástica (0,5)": 0.5, "lasso": 1.0}
F = {k: np.empty(n - i0) for k in M}; sel = np.zeros(len(VARS_MULT)); nsel = []; lams = {k: [] for k in M}
for a in anios:
    Xtr, ytr = X[:a], y[:a]; Xte = X[a:min(a + 12, n)]; k0 = a - i0
    S, St = estandarizar(Xtr, Xte); my = ytr.mean()
    Z = np.column_stack([np.ones(a), S]); b = np.linalg.solve(Z.T @ Z, Z.T @ ytr); F["MCO (todos)"][k0:k0 + len(Xte)] = b[0] + St @ b[1:]
    for nom, al in M.items():
        if al is None: continue
        lam, bb = ajustar_validado(Xtr, ytr, al); lams[nom].append(lam); F[nom][k0:k0 + len(Xte)] = my + St @ bb
        if nom == "lasso": sel += bb != 0; nsel.append((bb != 0).sum())
print(f"Predictores: {', '.join(VARS_MULT)} (de = dp - ep y tms = lty - tbl se omiten por colinealidad exacta)")
print(f"Fuera de muestra {mes[i0]} a {mes[-1]}; reestimación anual ({len(anios)} veces); lambda elegido con los últimos 120 meses de cada muestra")
print("R2 fuera (%)   R2 fuera CT (%)    CW     p(CW)   lambda mediano   modelo")
for nom in M:
    cw, p = clark_west(yo, F[nom], fb)
    lm = f"{np.median(lams[nom]):.4f}" if lams[nom] else "  -"
    print(f"{100 * r2_os(yo, F[nom], fb):11.2f}  {100 * r2_os(yo, np.maximum(F[nom], 0), fb):14.2f}  {cw:7.2f}  {p:7.4f}   {lm:>12s}   {nom}")
print(f"\nLasso: número medio de predictores seleccionados = {np.mean(nsel):.2f}; reestimaciones sin ningún predictor = {sum(1 for s in nsel if s == 0)} de {len(nsel)}")
print("fracción de reestimaciones en que el lasso selecciona cada predictor:")
print("  " + "  ".join(f"{v}={s / len(anios):.2f}" for v, s in zip(VARS_MULT, sel)))
print(f"\ndesviación estándar de los pronósticos: media histórica = {fb.std(ddof=1):.4f}; MCO = {F['MCO (todos)'].std(ddof=1):.4f}; ridge = {F['ridge'].std(ddof=1):.4f}; lasso = {F['lasso'].std(ddof=1):.4f}")
np.save("salidas/cap20_pronosticos_contraccion.npy", np.column_stack([F[k] for k in M]))

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(8.6, 5.0), sharex=True); t = np.arange(n - i0)
for nom, c, ls in [("MCO (todos)", eg.GRIS, "-"), ("ridge", eg.AZUL, "--"), ("lasso", eg.LADRILLO, "-")]:
    ax[0].plot(t, F[nom], color=c, ls=ls, lw=0.9, label=nom); ax[1].plot(t, np.cumsum((yo - fb)**2 - (yo - F[nom])**2), color=c, ls=ls, lw=1.2, label=nom)
ax[0].plot(t, fb, color="black", lw=0.8, label="media histórica"); ax[0].set_ylabel("pronóstico (%)"); ax[0].legend(fontsize=7, ncol=4)
ax[1].axhline(0, color="black", lw=0.5); ax[1].set_ylabel("SCE(media) - SCE(modelo)")
pos = [k for k in t if mes[i0 + k].endswith("-01") and int(mes[i0 + k][:4]) % 10 == 0]; ax[1].set_xticks(pos)
for a_ in ax: eg.coma(a_)
ax[1].set_xticklabels([mes[i0 + k][:4] for k in pos]); fig.savefig("graficos/cap20_ej02_contraccion.pdf")
