# Capítulo 19, Ejercicio 19.1: la ecuación de inversión de Grunfeld con MCO agrupado, between, efectos fijos y aleatorios
import numpy as np
from scipy.stats import chi2, f as fdist
from cap19_funciones import grunfeld, mco, media_grupo, v_clasica, efectos_aleatorios

g, t, y, X = grunfeld(); n, K = X.shape; N = len(np.unique(g)); T = n // N
print(f"Panel balanceado: N = {N} empresas, T = {T} años ({t.min()}-{t.max()}), n = {n}")
print(f"medias: inversión = {y.mean():.2f}; valor = {X[:, 0].mean():.2f}; capital = {X[:, 1].mean():.2f}")
def fila(nom, b, V, j0=0):
    ee = np.sqrt(np.diag(V)); print(f"  {nom:22s} valor {b[j0]:8.4f} ({ee[j0]:.4f})   capital {b[j0 + 1]:8.4f} ({ee[j0 + 1]:.4f})")
Z = np.column_stack([np.ones(n), X]); bp, ep, Ap = mco(y, Z); fila("MCO agrupado", bp, v_clasica(Z, ep, Ap, n - K - 1), 1)
_, idx = np.unique(g, return_index=True); yb, Xb = media_grupo(y, g)[idx], media_grupo(X, g)[idx]
Zb = np.column_stack([np.ones(N), Xb]); bb, eb, Ab = mco(yb, Zb); fila("between", bb, v_clasica(Zb, eb, Ab, N - K - 1), 1)
yw, Xw = y - media_grupo(y, g), X - media_grupo(X, g); bw, ew, Aw = mco(yw, Xw); Vw = v_clasica(Xw, ew, Aw, n - N - K); fila("efectos fijos (within)", bw, Vw)
bre, Vre, th, s2e, s2u = efectos_aleatorios(y, X, g); fila("efectos aleatorios", bre, Vre, 1)
orden = np.lexsort((t, g)); gg, yy, XX = g[orden], y[orden], X[orden]; m = gg[1:] == gg[:-1]
dy, dX = (yy[1:] - yy[:-1])[m], (XX[1:] - XX[:-1])[m]; bd, ed, Ad = mco(dy, dX); fila("primeras diferencias", bd, v_clasica(dX, ed, Ad, len(dy) - K))
print(f"\nefectos aleatorios: sigma_u = {np.sqrt(s2u):.4f}; sigma_e = {np.sqrt(s2e):.4f}; theta = {th:.4f}; constante = {bre[0]:.4f}")
alfa = (media_grupo(y, g) - media_grupo(X, g) @ bw)[idx]
print("efectos fijos estimados por empresa: " + " ".join(f"{a:.1f}" for a in alfa))
Fe = ((ep @ ep - ew @ ew) / (N - 1)) / ((ew @ ew) / (n - N - K))
print(f"F de efectos individuales = {Fe:.4f} con ({N - 1}, {n - N - K}) gl; valor p = {fdist.sf(Fe, N - 1, n - N - K):.2e}")
d = bw - bre[1:]; H = d @ np.linalg.solve(Vw - Vre[1:, 1:], d)
print(f"Hausman (efectos fijos frente a aleatorios) = {H:.4f}; valor p chi2(2) = {chi2.sf(H, 2):.4f}")
r2w = 1 - (ew @ ew) / (yw @ yw); print(f"R2 within = {r2w:.4f}; R2 del MCO agrupado = {1 - (ep @ ep) / ((y - y.mean()) @ (y - y.mean())):.4f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.6, 4.0)); cols = plt.cm.tab10.colors if False else None
for k, c in enumerate(np.unique(g)):
    s = g == c; ax.plot(X[s, 0], y[s], "o", ms=3, color=[eg.AZUL, eg.LADRILLO, eg.OLIVA, eg.GRIS][k % 4], alpha=0.7)
    xs = np.linspace(X[s, 0].min(), X[s, 0].max(), 50); yh = alfa[k] + bw[0] * xs + bw[1] * X[s, 1].mean(); ok_ = yh > 1; ax.plot(xs[ok_], yh[ok_], color=[eg.AZUL, eg.LADRILLO, eg.OLIVA, eg.GRIS][k % 4], lw=0.8)
xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 10); ax.plot(xs, bp[0] + bp[1] * xs + bp[2] * X[:, 1].mean(), color="black", ls="--", lw=1.2, label="MCO agrupado")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("valor de la empresa"); ax.set_ylabel("inversión"); ax.legend(fontsize=8); eg.coma(ax)
fig.savefig("graficos/cap19_ej01_grunfeld.pdf")
