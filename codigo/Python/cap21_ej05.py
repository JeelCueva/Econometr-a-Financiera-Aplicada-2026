# Capítulo 21, Etapa 5: asignación: 1/N frente a mínima varianza fuera de muestra
import numpy as np
from cap21_funciones import cargar, NOMBRES, min_varianza, max_caida

f, R = cargar(); A = R[:, :4]; n = len(A); V = 250; paso = 21
H = np.empty((n, 4, 4)); H[0] = A[:30].T @ A[:30] / 30
for t in range(1, n): H[t] = 0.94 * H[t - 1] + 0.06 * np.outer(A[t - 1], A[t - 1])
reb = list(range(V, n, paso)); W = {"1/N": [], "mínima varianza (muestral, 250 días)": [], "mínima varianza (EWMA)": []}
for t in reb:
    W["1/N"].append(np.full(4, 0.25)); W["mínima varianza (muestral, 250 días)"].append(min_varianza(np.cov(A[t - V:t].T))); W["mínima varianza (EWMA)"].append(min_varianza(H[t]))
print(f"Evaluación: {f[V]} a {f[-1]} ({n - V} días); rebalanceo cada {paso} días ({len(reb)} veces); pesos constantes entre rebalanceos")
print("\nmedia anual (%)   vol. anual (%)   Sharpe anual   ee(Sharpe)   máx. caída (%)   rotación media   peso mínimo medio   estrategia")
RP = {}
for nom, Ws in W.items():
    Ws = np.array(Ws); rp = np.empty(n - V)
    for k, t in enumerate(reb): fin = min(t + paso, n); rp[t - V:fin - V] = A[t:fin] @ Ws[k]
    RP[nom] = rp; sr = rp.mean() / rp.std(ddof=1); T = len(rp); ee = np.sqrt((1 + sr**2 / 2) / T) * np.sqrt(252)
    rot = np.abs(np.diff(Ws, axis=0)).sum(1).mean()
    print(f"{252 * rp.mean():14.2f}  {np.sqrt(252) * rp.std(ddof=1):15.2f}  {np.sqrt(252) * sr:13.3f}  {ee:11.3f}  {max_caida(rp):14.2f}  {rot:14.3f}  {Ws.min(1).mean():17.3f}   {nom}")
Wm = np.array(W["mínima varianza (muestral, 250 días)"])
print("\npesos medios de mínima varianza (muestral): " + "  ".join(f"{a} = {w:.3f}" for a, w in zip(NOMBRES[:4], Wm.mean(0))))
print(f"rebalanceos con alguna posición corta: {(Wm.min(1) < 0).sum()} de {len(reb)}")
d = RP["mínima varianza (muestral, 250 días)"] - RP["1/N"]
print(f"diferencia de varianzas diarias (mínima varianza - 1/N) = {RP['mínima varianza (muestral, 250 días)'].var(ddof=1) - RP['1/N'].var(ddof=1):.4f}; correlación entre ambas estrategias = {np.corrcoef(RP['1/N'], RP['mínima varianza (muestral, 250 días)'])[0, 1]:.4f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [1.5, 1]}); t = np.arange(V, n)
for (nom, rp), c, ls in zip(RP.items(), [eg.GRIS, eg.AZUL, eg.LADRILLO], ["-", "--", ":"]): ax[0].plot(t, 100 * np.cumprod(1 + rp / 100), color=c, ls=ls, lw=1.2, label=nom)
pos = [k for k in t if f[k][:4] != f[k - 1][:4]]; ax[0].set_xticks(pos); ax[0].set_ylabel("valor de 100 invertidos"); ax[0].legend(fontsize=7)
ax[1].stackplot(np.array(reb), np.clip(Wm, 0, None).T, colors=[eg.AZUL, eg.LADRILLO, eg.OLIVA, eg.GRIS], alpha=0.8, labels=NOMBRES[:4])
ax[1].set_xticks(pos); ax[1].set_title("Pesos de mínima varianza (muestral)", fontsize=9); ax[1].legend(fontsize=7, loc="upper left", ncol=2)
for a in ax: eg.coma(a)
for a in ax: a.set_xticklabels([f[k][:4] for k in pos])
fig.tight_layout(); fig.savefig("graficos/cap21_ej05_asignacion.pdf")
