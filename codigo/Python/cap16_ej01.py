# Capítulo 16, Ejercicio 16.1: Facebook y su anuncio de resultados del 25 de julio de 2018
import numpy as np
from scipy.stats import t as tdist
from cap16_funciones import cargar, rendimientos_normales, car, ACT

f, R = cargar(); fb, m = R[:, 2], R[:, 4]
t0 = int(np.where(f == "2018-07-26")[0][0]); print(f"día 0 = {f[t0]} (posición {t0}); ventana de estimación: {f[t0 - 250]} a {f[t0 - 11]}")
print(f"rendimiento de FB el día 0 = {fb[t0]:.4f} %; del S&P 500 = {m[t0]:.4f} %\n")
print("modelo      alfa      beta      s     AR(0)    ee(AR0)   t(AR0)   CAR(-1,+1)  ee     CAR(0,+5)   ee     CAR(-5,+5)   ee")
res = {}
for mod in ["mercado", "media", "ajustado"]:
    r = rendimientos_normales(fb, m, t0, modelo=mod); res[mod] = r; ee0 = np.sqrt(r["V"][5, 5])
    a = r["b"][0] if len(r["b"]) else 0.0; b = r["b"][1] if len(r["b"]) > 1 else (1.0 if mod == "ajustado" else 0.0)
    c1, c2, c3 = car(r, -1, 1), car(r, 0, 5), car(r, -5, 5)
    print(f"{mod:9s} {a:8.4f} {b:8.4f} {np.sqrt(r['s2']):7.4f} {r['AR'][5]:9.4f} {ee0:8.4f} {r['AR'][5] / ee0:8.2f}  "
          f"{c1[0]:9.4f} {c1[1]:7.4f} {c2[0]:9.4f} {c2[1]:7.4f} {c3[0]:9.4f} {c3[1]:7.4f}")
r = res["mercado"]
print("\nModelo de mercado, ventana del evento:")
print("  tau   fecha        R_FB      R_m       AR      ee(AR)     t      CAR")
for k, tau in enumerate(r["tau"]):
    print(f"  {tau:3d}  {f[t0 + tau]}  {fb[t0 + tau]:8.4f} {m[t0 + tau]:8.4f} {r['AR'][k]:8.4f} {np.sqrt(r['V'][k, k]):8.4f} {r['AR'][k] / np.sqrt(r['V'][k, k]):7.2f} {r['AR'][:k + 1].sum():8.4f}")
print(f"  valor crítico t({r['gl']}) al 5 % bilateral = {tdist.ppf(0.975, r['gl']):.4f}")
# Equivalencia con una variable ficticia: regresión en la ventana de estimación más el día 0
ie = np.r_[np.arange(t0 - 250, t0 - 10), t0]; X = np.column_stack([np.ones(len(ie)), m[ie], (ie == t0).astype(float)])
b = np.linalg.solve(X.T @ X, X.T @ fb[ie]); e = fb[ie] - X @ b; s2 = (e @ e) / (len(ie) - 3)
ee = np.sqrt(s2 * np.linalg.inv(X.T @ X)[2, 2])
print(f"\nRegresión con ficticia del día 0: delta = {b[2]:.4f}; ee = {ee:.4f}; t = {b[2] / ee:.2f}")
# Sensibilidad: excluir de la estimación los días 0 a +5 del caso Cambridge Analytica (19 de marzo de 2018)
tc = int(np.where(f == "2018-03-19")[0][0]); rx = rendimientos_normales(fb, m, t0, excluir=range(tc, tc + 6))
print(f"Excluyendo {f[tc]} a {f[tc + 5]} de la estimación: beta = {rx['b'][1]:.4f}; s = {np.sqrt(rx['s2']):.4f}; "
      f"AR(0) = {rx['AR'][5]:.4f}; t = {rx['AR'][5] / np.sqrt(rx['V'][5, 5]):.2f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.3)); tau = r["tau"]; ee = np.sqrt(np.diag(r["V"]))
ax[0].bar(tau, r["AR"], color=[eg.LADRILLO if k == 0 else eg.AZUL for k in tau], width=0.7)
ax[0].fill_between(tau, -1.96 * ee, 1.96 * ee, color=eg.GRIS, alpha=0.25, label="$\\pm1{,}96$ ee")
ax[0].set_xlabel("$\\tau$ (días hábiles)"); ax[0].set_ylabel("AR (%)"); ax[0].set_title("Rendimientos anormales", fontsize=9); ax[0].legend(fontsize=8)
cs = np.cumsum(r["AR"]); ec = np.array([car(r, -5, k)[1] for k in tau])
ax[1].plot(tau, cs, "o-", color=eg.AZUL, ms=3, label="CAR$(-5,\\tau)$"); ax[1].fill_between(tau, cs - 1.96 * ec, cs + 1.96 * ec, color=eg.AZUL, alpha=0.15, label="banda al 95 %")
ax[1].axhline(0, color="black", lw=0.5); ax[1].set_xlabel("$\\tau$ (días hábiles)"); ax[1].set_ylabel("CAR (%)"); ax[1].set_title("Rendimiento anormal acumulado", fontsize=9); ax[1].legend(fontsize=8)
for a in ax: eg.coma(a)
fig.savefig("graficos/cap16_ej01_facebook.pdf")
