# Capítulo 17, Ejercicio 17.3: el ciclo de la producción industrial de EE. UU. con dos especificaciones
import numpy as np
from cap17_funciones import produccion, em_msiar, em_media_varianza, ergodicas, recesion_trimestral

q, y = produccion(); T = len(y); rec = recesion_trimestral(q)
print(f"Producción industrial, variación trimestral (%), {q[0]} a {q[-1]}; T = {T}; trimestres de recesión NBER: {rec.sum()}")
print(f"media = {y.mean():.4f}; desv. est. = {y.std(ddof=1):.4f}; autocorrelación(1) = {np.corrcoef(y[1:], y[:-1])[0, 1]:.4f}")
X = np.column_stack([np.ones(T - 1), y[:-1]]); b = np.linalg.lstsq(X, y[1:], rcond=None)[0]; e = y[1:] - X @ b; s2 = e @ e / (T - 1)
print(f"AR(1) lineal: c = {b[0]:.4f}; phi = {b[1]:.4f}; sigma = {np.sqrt(s2):.4f}; log-verosimilitud = {-0.5 * (T - 1) * (np.log(2 * np.pi * s2) + 1):.4f}")

def resumen(p1, xf, r_ind, qq):
    c = (p1 > 0.5).astype(int)
    print(f"  trimestres con Pr suavizada de recesión > 0,5: {c.sum()}; coinciden con el NBER: {((c == 1) & (r_ind == 1)).sum()}; "
          f"falsas alarmas: {((c == 1) & (r_ind == 0)).sum()}; no detectados: {((c == 0) & (r_ind == 1)).sum()}")
    print(f"  concordancia = {(c == r_ind).mean():.4f}; correlación con el indicador NBER = {np.corrcoef(p1, r_ind)[0, 1]:.4f}")
    print("  trimestres con Pr filtrada > 0,5: " + " ".join(qq[xf > 0.5]))

A = em_msiar(y, 2, [-1.5, 1.0], 0.3, y.std(), [[0.8, 0.2], [0.1, 0.9]]); P = A["P"]; pi = ergodicas(P)
print(f"\n(A) MSI-AR(1), varianza común: {A['it']} iteraciones; log-verosimilitud = {A['ll']:.4f} (condicionada en y_1)")
print(f"  c1 = {A['c'][0]:.4f}; c2 = {A['c'][1]:.4f}; phi = {A['phi']:.4f}; sigma = {A['s']:.4f}")
print(f"  p11 = {P[0, 0]:.4f}; p22 = {P[1, 1]:.4f}; duraciones = {1 / (1 - P[0, 0]):.2f} y {1 / (1 - P[1, 1]):.2f} trimestres; ergódicas = {pi[0]:.4f} {pi[1]:.4f}")
resumen(A["xs"][:, 0], A["xf"][:, 0], rec[1:], q[1:])
B = em_media_varianza(y, 2, [-1.0, 1.0], [1.5, 1.0], [[0.8, 0.2], [0.1, 0.9]]); P = B["P"]; pi = ergodicas(P)
print(f"\n(B) Media y varianza cambiantes: {B['it']} iteraciones; log-verosimilitud = {B['ll']:.4f}")
print(f"  mu1 = {B['mu'][0]:.4f}; mu2 = {B['mu'][1]:.4f}; sigma1 = {B['s'][0]:.4f}; sigma2 = {B['s'][1]:.4f}")
print(f"  p11 = {P[0, 0]:.4f}; p22 = {P[1, 1]:.4f}; duraciones = {1 / (1 - P[0, 0]):.2f} y {1 / (1 - P[1, 1]):.2f} trimestres; ergódicas = {pi[0]:.4f} {pi[1]:.4f}")
resumen(B["xs"][:, 0], B["xf"][:, 0], rec, q)
z = y - B["xs"] @ B["mu"]
print(f"  autocorrelación(1) de y_t menos la media suavizada del régimen = {np.corrcoef(z[1:], z[:-1])[0, 1]:.4f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
t = np.arange(T); pos = [k for k in range(T) if q[k].endswith("T1") and int(q[k][:4]) % 5 == 0]
fig, ax = plt.subplots(3, 1, figsize=(8.6, 6.0), sharex=True)
for a in ax:
    for k in range(T):
        if rec[k]: a.axvspan(k - 0.5, k + 0.5, color=eg.GRIS, alpha=0.35, lw=0)
ax[0].plot(t, y, color=eg.AZUL, lw=0.9); ax[0].axhline(0, color="black", lw=0.5); ax[0].set_ylabel("variación (%)")
ax[1].plot(t[1:], A["xs"][:, 0], color=eg.LADRILLO, lw=1.0); ax[1].set_ylabel("Pr(rég. 1), (A)"); ax[1].set_title("(A) MSI-AR(1) con varianza común", fontsize=9)
ax[2].plot(t, B["xs"][:, 0], color=eg.LADRILLO, lw=1.0); ax[2].set_ylabel("Pr(rég. 1), (B)"); ax[2].set_title("(B) media y varianza cambiantes", fontsize=9)
ax[2].set_xticks(pos); ax[2].set_xticklabels([q[k][:4] for k in pos])
for a in ax: eg.coma(a)
ax[2].set_xticklabels([q[k][:4] for k in pos]); fig.savefig("graficos/cap17_ej03_produccion.pdf")
