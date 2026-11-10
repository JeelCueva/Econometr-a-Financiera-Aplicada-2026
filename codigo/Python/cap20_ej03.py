# Capítulo 20, Ejercicio 20.3: combinación de pronósticos, componentes principales y valor económico
import numpy as np
from cap20_funciones import cargar, VARS, IDX_MULT, estandarizar, r2_os, clark_west

mes, X, y = cargar(); n = len(y); i0 = int(np.where(mes == "1957-01")[0][0]); yo = y[i0:]
fb = np.array([y[:i].mean() for i in range(i0, n)])
U = np.empty((n - i0, len(VARS)))
for j in range(len(VARS)):
    for k, i in enumerate(range(i0, n)):
        Z = np.column_stack([np.ones(i), X[:i, j]]); b = np.linalg.solve(Z.T @ Z, Z.T @ y[:i]); U[k, j] = b[0] + b[1] * X[i, j]
F = {"combinación (media de 10)": U.mean(1), "combinación (mediana de 10)": np.median(U, 1)}
Xm = X[:, IDX_MULT]
for q in (1, 2, 3):
    f = np.empty(n - i0)
    for a in range(i0, n, 12):
        S, St = estandarizar(Xm[:a], Xm[a:min(a + 12, n)]); w, V = np.linalg.eigh(S.T @ S / a); V = V[:, ::-1][:, :q]
        P, Pt = S @ V, St @ V; Z = np.column_stack([np.ones(a), P]); b = np.linalg.solve(Z.T @ Z, Z.T @ y[:a]); f[a - i0:a - i0 + len(St)] = b[0] + Pt @ b[1:]
    F[f"componentes principales (k = {q})"] = f
print(f"Fuera de muestra {mes[i0]} a {mes[-1]} ({n - i0} meses)")
print("R2 fuera (%)    CW     p(CW)   modelo")
for nom, f in F.items():
    cw, p = clark_west(yo, f, fb); print(f"{100 * r2_os(yo, f, fb):11.2f}  {cw:6.2f}  {p:7.4f}   {nom}")
print(f"\ndesviación estándar de los pronósticos: combinación = {F['combinación (media de 10)'].std(ddof=1):.4f}; media histórica = {fb.std(ddof=1):.4f}")
print(f"correlación entre la combinación y el exceso realizado = {np.corrcoef(F['combinación (media de 10)'], yo)[0, 1]:.4f}")
# Valor económico (Campbell y Thompson, 2008): inversionista media-varianza con aversión al riesgo 3
gam = 3.0; s2 = np.array([y[i - 60:i].var(ddof=1) for i in range(i0, n)]); rf = X[i0:, VARS.index("tbl")] / 12
def eqc(f):
    w = np.clip(100 * f / (gam * s2), 0, 1.5); rp = rf + w * yo; return 12 * (rp.mean() - 0.5 * gam * rp.var(ddof=1) / 100), w
ce0, w0 = eqc(fb)
print(f"\nInversionista media-varianza (aversión 3, varianza de 60 meses, peso en acciones entre 0 y 150 %):")
print(f"  equivalente cierto anual con la media histórica = {ce0:.3f} %; peso medio en acciones = {w0.mean():.3f}")
for nom in ["combinación (media de 10)", "componentes principales (k = 1)"]:
    ce, w = eqc(F[nom]); print(f"  {nom}: equivalente cierto = {ce:.3f} %; ganancia = {ce - ce0:.3f} puntos por año; peso medio = {w.mean():.3f}")
ce, w = eqc(U[:, VARS.index("dp")]); print(f"  solo dp: equivalente cierto = {ce:.3f} %; ganancia = {ce - ce0:.3f} puntos por año; peso medio = {w.mean():.3f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8.6, 3.2)); t = np.arange(n - i0)
for nom, c, ls in [("combinación (media de 10)", eg.AZUL, "-"), ("componentes principales (k = 1)", eg.LADRILLO, "--"), ("componentes principales (k = 3)", eg.GRIS, ":")]:
    ax.plot(t, np.cumsum((yo - fb)**2 - (yo - F[nom])**2), color=c, ls=ls, lw=1.2, label=nom)
ax.axhline(0, color="black", lw=0.5); pos = [k for k in t if mes[i0 + k].endswith("-01") and int(mes[i0 + k][:4]) % 10 == 0]
ax.set_xticks(pos); ax.set_ylabel("SCE(media) - SCE(modelo)"); ax.legend(fontsize=8); eg.coma(ax); ax.set_xticklabels([mes[i0 + k][:4] for k in pos])
fig.savefig("graficos/cap20_ej03_combinacion.pdf")
