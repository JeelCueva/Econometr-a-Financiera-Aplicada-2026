# Capítulo 7, Ejercicio 7.2: portafolios pequeños frente a grandes; efecto adelanto-rezago
import numpy as np, pandas as pd, statsmodels.api as sm
from cap07_funciones import vr_lm

f = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv")
print(f"Muestra: {f['dates'].iloc[0]} a {f['dates'].iloc[-1]}, T = {len(f)} meses\n")
print("Cartera    rho1   VR(2)  z*(2)   VR(4)  z*(4)  VR(12) z*(12)")
for p in ["S1V1", "S1V3", "S1V5", "S3V3", "S5V1", "S5V3", "S5V5"]:
    x = f[p].to_numpy()
    rho1 = ((x[1:] - x.mean()) * (x[:-1] - x.mean())).sum() / ((x - x.mean())**2).sum()  # ACF muestral
    v = [vr_lm(x, q) for q in (2, 4, 12)]
    print(f"{p:<8} {rho1:6.3f}  " + " ".join(f"{d['VR']:6.2f} {d['zrob']:6.2f}" for d in v))
pe, gr = f["S1V3"].to_numpy(), f["S5V3"].to_numpy(); n = len(pe)
print("\nCorrelaciones cruzadas (pequeña = S1V3, grande = S5V3):")
for k in range(4):
    a = np.corrcoef(pe[k:], gr[:n - k])[0, 1]; b = np.corrcoef(gr[k:], pe[:n - k])[0, 1]
    print(f"  k = {k}: corr(peq_t, gran_t-k) = {a:6.3f}   corr(gran_t, peq_t-k) = {b:6.3f}")
X = sm.add_constant(np.column_stack([pe[:-1], gr[:-1]]))
m = sm.OLS(pe[1:], X).fit()
print("\nRegresión peq_t = a + b1 peq_t-1 + b2 gran_t-1:")
print(pd.DataFrame({"coef": m.params, "ee": m.bse, "t": m.tvalues, "p": m.pvalues},
                   index=["a", "b1 (peq_t-1)", "b2 (gran_t-1)"]).round(4))

# ---- Figura: correlaciones cruzadas asimétricas ----
import estilo_graficos as eg, matplotlib.pyplot as plt
ks = np.arange(1, 7)
ca = [np.corrcoef(pe[k:], gr[:n - k])[0, 1] for k in ks]; cb = [np.corrcoef(gr[k:], pe[:n - k])[0, 1] for k in ks]
fig, ax = plt.subplots(figsize=(6.2, 3.2)); w = 0.36
ax.bar(ks - w / 2, ca, w, color=eg.LADRILLO, label="corr(pequeña$_t$, grande$_{t-k}$)")
ax.bar(ks + w / 2, cb, w, color=eg.AZUL, label="corr(grande$_t$, pequeña$_{t-k}$)")
ax.axhline(1.96 / np.sqrt(n), color=eg.GRIS, ls=":"); ax.axhline(-1.96 / np.sqrt(n), color=eg.GRIS, ls=":")
ax.axhline(0, color="black", lw=0.6); ax.set_xlabel("rezago $k$ (meses)"); ax.set_ylabel("correlación")
ax.legend(fontsize=8); eg.coma(ax); fig.savefig("graficos/cap07_ej02_adelanto_rezago.pdf")
