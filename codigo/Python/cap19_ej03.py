# Capítulo 19, Ejercicio 19.3: paneles dinámicos: el sesgo de Nickell y el estimador de Anderson-Hsiao
import numpy as np
from cap19_funciones import grunfeld, mco, media_grupo, mc2e, rezagar, v_clasica, LCG

g, t, inv, X = grunfeld(); y = np.log(inv); x = np.log(X[:, 0])
y1, y2, x1 = rezagar(y, g, t, 1), rezagar(y, g, t, 2), rezagar(x, g, t, 1)
print("Modelo: ln I_it = rho ln I_i,t-1 + beta ln F_it + alfa_i + e_it  (Grunfeld, 1935-1954)")
ok = ~np.isnan(y1); n1 = ok.sum()
Z = np.column_stack([np.ones(n1), y1[ok], x[ok]]); b, e, A = mco(y[ok], Z); s = np.sqrt(np.diag(v_clasica(Z, e, A, n1 - 3)))
print(f"  MCO agrupado:        rho = {b[1]:.4f} ({s[1]:.4f}); beta = {b[2]:.4f} ({s[2]:.4f}); n = {n1}")
gg = g[ok]; N = len(np.unique(gg))
yw = y[ok] - media_grupo(y[ok], gg); Xw = np.column_stack([y1[ok], x[ok]]) - media_grupo(np.column_stack([y1[ok], x[ok]]), gg)
bw, ew, Aw = mco(yw, Xw); sw = np.sqrt(np.diag(v_clasica(Xw, ew, Aw, n1 - N - 2)))
print(f"  efectos fijos:       rho = {bw[0]:.4f} ({sw[0]:.4f}); beta = {bw[1]:.4f} ({sw[1]:.4f})")
ok2 = ~np.isnan(y2); dy, dy1, dx = (y - y1)[ok2], (y1 - y2)[ok2], (x - x1)[ok2]
bah, Vah = mc2e(dy, np.column_stack([dy1, dx]), np.column_stack([y2[ok2], dx])); sah = np.sqrt(np.diag(Vah))
print(f"  Anderson-Hsiao (VI): rho = {bah[0]:.4f} ({sah[0]:.4f}); beta = {bah[1]:.4f} ({sah[1]:.4f}); n = {ok2.sum()}; instrumento: ln I_i,t-2")
r1 = np.corrcoef(y2[ok2], dy1)[0, 1]; print(f"  correlación entre el instrumento y el regresor endógeno = {r1:.4f}")
T = 19; print(f"\nSesgo aproximado de Nickell con T = {T} y rho = {bw[0]:.2f}: -(1 + rho)/(T - 1) = {-(1 + bw[0]) / (T - 1):.4f}")
# Simulación del sesgo con el tamaño de este panel: N = 10, T = 20, rho = 0,7, efectos fijos normales
G = LCG(2019); REP, N, T, rho = 1000, 10, 20, 0.7; est = []
for r in range(REP):
    a = np.array([G.normal() for _ in range(N)]); Y = np.zeros((N, T + 50))
    for i in range(N):
        Y[i, 0] = a[i] / (1 - rho)
        for k in range(1, T + 50): Y[i, k] = a[i] + rho * Y[i, k - 1] + G.normal()
    Y = Y[:, 50:]; yy, yl = Y[:, 1:], Y[:, :-1]
    fe = ((yy - yy.mean(1, keepdims=True)) * (yl - yl.mean(1, keepdims=True))).sum() / ((yl - yl.mean(1, keepdims=True))**2).sum()
    d, dl, l2 = (Y[:, 2:] - Y[:, 1:-1]).ravel(), (Y[:, 1:-1] - Y[:, :-2]).ravel(), Y[:, :-2].ravel()
    est.append((fe, (l2 @ d) / (l2 @ dl)))
est = np.array(est)
print(f"Simulación ({REP} paneles con N = {N}, T = {T}, rho = {rho}):")
print(f"  efectos fijos:  media = {est[:, 0].mean():.4f}; sesgo = {est[:, 0].mean() - rho:.4f}; desv. est. = {est[:, 0].std(ddof=1):.4f}; fórmula -(1+rho)/(T-1) = {-(1 + rho) / (T - 2):.4f}")
print(f"  Anderson-Hsiao: mediana = {np.median(est[:, 1]):.4f}; sesgo de la mediana = {np.median(est[:, 1]) - rho:.4f}; rango intercuartílico = {np.quantile(est[:, 1], 0.75) - np.quantile(est[:, 1], 0.25):.4f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 3.3)); bins = np.linspace(0, 1.2, 49)
ax.hist(est[:, 0], bins=bins, color=eg.AZUL, alpha=0.6, label="efectos fijos"); ax.hist(np.clip(est[:, 1], 0, 1.2), bins=bins, color=eg.LADRILLO, alpha=0.45, label="Anderson-Hsiao (recortado a [0; 1,2])")
ax.axvline(rho, color="black", lw=1.2, ls="--", label="$\\rho$ verdadero = 0,7"); ax.set_xlabel("$\\hat\\rho$"); ax.set_ylabel("frecuencia"); ax.legend(fontsize=8); eg.coma(ax)
fig.savefig("graficos/cap19_ej03_nickell.pdf")
