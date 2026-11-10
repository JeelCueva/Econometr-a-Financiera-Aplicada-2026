# Capítulo 19, Ejercicio 19.2: errores estándar en paneles: robustos, agrupados y de Driscoll-Kraay (Grunfeld)
import numpy as np
from scipy.stats import t as tdist
from cap19_funciones import grunfeld, mco, media_grupo, v_clasica, v_white, v_cluster, v_dos_vias, v_driscoll_kraay

g, t, y, X = grunfeld(); n, K = X.shape; N = len(np.unique(g)); T = n // N
yw, Xw = y - media_grupo(y, g), X - media_grupo(X, g); b, e, A = mco(yw, Xw)
# correlación de los residuos dentro de cada empresa (en el tiempo) y dentro de cada año (entre empresas)
E = np.array([e[g == c][np.argsort(t[g == c])] for c in np.unique(g)])      # N x T
rho_t = np.mean([np.corrcoef(E[i, 1:], E[i, :-1])[0, 1] for i in range(N)])
C = np.corrcoef(E); rho_c = C[np.triu_indices(N, 1)].mean()
print(f"Efectos fijos de empresa: valor = {b[0]:.4f}; capital = {b[1]:.4f}")
print(f"autocorrelación media de orden 1 de los residuos de cada empresa = {rho_t:.4f}")
print(f"correlación media entre los residuos de dos empresas en el mismo año = {rho_c:.4f}")
V = {"clásicos": v_clasica(Xw, e, A, n - N - K), "White (HC1)": v_white(Xw, e, A, K + N),
     "agrupados por empresa": v_cluster(Xw, e, A, g, K + N), "agrupados por año": v_cluster(Xw, e, A, t, K + N),
     "dos vías": v_dos_vias(Xw, e, A, g, t, K + N), "Driscoll-Kraay (L = 2)": v_driscoll_kraay(Xw, e, A, t, 2)}
print("\n  valor     capital    t(valor)  t(capital)   error estándar")
for nom, Vm in V.items():
    s = np.sqrt(np.diag(Vm)); print(f"  {s[0]:.4f}    {s[1]:.4f}    {b[0] / s[0]:7.2f}   {b[1] / s[1]:7.2f}      {nom}")
print(f"\nvalor crítico al 5 %: normal = 1,96; t con G - 1 = {N - 1} gl = {tdist.ppf(0.975, N - 1):.4f}")
# efectos fijos en dos vías (empresa y año), panel balanceado
y2 = y - media_grupo(y, g) - media_grupo(y, t) + y.mean(); X2 = X - media_grupo(X, g) - media_grupo(X, t) + X.mean(0)
b2, e2, A2 = mco(y2, X2); s2 = np.sqrt(np.diag(v_clasica(X2, e2, A2, n - N - T + 1 - K))); s2c = np.sqrt(np.diag(v_cluster(X2, e2, A2, g, K + N + T - 1)))
print(f"\nEfectos fijos de empresa y año: valor = {b2[0]:.4f} (clásico {s2[0]:.4f}; agrupado {s2c[0]:.4f}); capital = {b2[1]:.4f} (clásico {s2[1]:.4f}; agrupado {s2c[1]:.4f})")
print(f"efectos de año (desviaciones de la media): " + " ".join(f"{v:.1f}" for v in (media_grupo(y - X @ b2, t) - (y - X @ b2).mean())[np.unique(t, return_index=True)[1]]))
