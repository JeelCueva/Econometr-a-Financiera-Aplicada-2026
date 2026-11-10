# Capítulo 20, Ejercicio 20.4: clasificación: la pendiente de la curva y las recesiones (Estrella y Mishkin, 1998)
import numpy as np, pandas as pd
from scipy.stats import chi2
from cap20_funciones import logit, auc

d = pd.read_csv("datos/internacional/tasas_cupon_cero_eeuu_mk.csv"); f = d["fecha"].to_numpy(); r = pd.read_csv("datos/internacional/recesiones_nber.csv")
rec = np.zeros(len(f), int)
for p, v in zip(r["pico"], r["valle"]): rec[(f > p) & (f <= v)] = 1
s = (d["r120"] - d["r3"]).to_numpy(); n3 = d["r3"].to_numpy()
print(f"Datos mensuales {f[0]} a {f[-1]}; meses en recesión: {rec.sum()} de {len(f)}")
print("\nhorizonte h   b0        b1 (pendiente)      ee(b1)    pseudo R2   AUC      n")
for h in (3, 6, 12, 18, 24):
    yv, x = rec[h:], s[:-h]; X = np.column_stack([np.ones(len(x)), x]); b, V, ll = logit(yv, X)
    p0 = yv.mean(); ll0 = len(yv) * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    print(f"   {h:3d}     {b[0]:8.4f}   {b[1]:8.4f}          {np.sqrt(V[1, 1]):.4f}    {1 - ll / ll0:8.4f}   {auc(yv, X @ b):.4f}   {len(yv)}")
h = 12; yv, x, lv = rec[h:], s[:-h], n3[:-h]; X1 = np.column_stack([np.ones(len(x)), x]); X2 = np.column_stack([X1, lv])
b1, V1, ll1 = logit(yv, X1); b2, V2, ll2 = logit(yv, X2)
print(f"\nh = 12: agregar el nivel de la tasa a 3 meses: b = ({b2[0]:.4f}, {b2[1]:.4f}, {b2[2]:.4f}); LR = {2 * (ll2 - ll1):.2f}; p = {chi2.sf(2 * (ll2 - ll1), 1):.4f}; AUC = {auc(yv, X2 @ b2):.4f}")
pr = 1 / (1 + np.exp(-X1 @ b1))
print(f"Pendiente que da Pr(recesión en 12 meses) = 50 %: {-b1[0] / b1[1]:.4f} puntos; con pendiente 0: {1 / (1 + np.exp(-b1[0])):.4f}; con pendiente 2: {1 / (1 + np.exp(-b1[0] - 2 * b1[1])):.4f}")
for u in (0.5, yv.mean()):
    c = (pr > u).astype(int); tp = ((c == 1) & (yv == 1)).sum(); fp = ((c == 1) & (yv == 0)).sum(); fn = ((c == 0) & (yv == 1)).sum(); tn = ((c == 0) & (yv == 0)).sum()
    print(f"  umbral {u:.4f}: verdaderos positivos = {tp}, falsos positivos = {fp}, falsos negativos = {fn}, verdaderos negativos = {tn}; sensibilidad = {tp / (tp + fn):.4f}; especificidad = {tn / (tn + fp):.4f}")
# fuera de muestra: logit con ventana creciente desde 1965
i0 = int(np.where(f[h:] == "1965-01")[0][0]); po = np.empty(len(yv) - i0)
for k, i in enumerate(range(i0, len(yv))):
    bb, _, _ = logit(yv[:i - h + 1], X1[:i - h + 1]); po[k] = 1 / (1 + np.exp(-X1[i] @ bb))   # solo resultados ya observados en t
print(f"\nFuera de muestra ({f[h:][i0]} a {f[h:][-1]}, solo con recesiones ya observadas): AUC = {auc(yv[i0:], po):.4f}; Pr media en meses de recesión = {po[yv[i0:] == 1].mean():.4f}; en expansión = {po[yv[i0:] == 0].mean():.4f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [1.7, 1]}); t = np.arange(len(yv))
for k in t:
    if yv[k]: ax[0].axvspan(k - 0.5, k + 0.5, color=eg.GRIS, alpha=0.35, lw=0)
ax[0].plot(t, pr, color=eg.LADRILLO, lw=1.0, label="Pr(recesión en $t+12$ | pendiente en $t$)")
pos = [k for k in t if f[h:][k].endswith("-01") and int(f[h:][k][:4]) % 5 == 0]; ax[0].set_xticks(pos); ax[0].legend(fontsize=7, loc="upper left")
ax[0].set_ylabel("probabilidad"); ax[0].set_title("Franjas: mes $t+12$ en recesión", fontsize=9)
o = np.argsort(-pr); tpr = np.cumsum(yv[o]) / yv.sum(); fpr = np.cumsum(1 - yv[o]) / (1 - yv).sum()
ax[1].plot(np.r_[0, fpr], np.r_[0, tpr], color=eg.AZUL, lw=1.4, label=f"AUC = {auc(yv, X1 @ b1):.3f}".replace(".", ",")); ax[1].plot([0, 1], [0, 1], color=eg.GRIS, ls=":")
ax[1].set_xlabel("tasa de falsos positivos"); ax[1].set_ylabel("tasa de verdaderos positivos"); ax[1].set_title("Curva ROC", fontsize=9); ax[1].legend(fontsize=8)
for a in ax: eg.coma(a)
ax[0].set_xticklabels([f[h:][k][:4] for k in pos]); fig.tight_layout(); fig.savefig("graficos/cap20_ej04_recesiones.pdf")
