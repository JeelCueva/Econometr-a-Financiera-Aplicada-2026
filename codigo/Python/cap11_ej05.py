# Capítulo 11, Ejercicio 11.5: ¿ayuda la corrección de errores a pronosticar? VECM frente a VAR en diferencias
import numpy as np, pandas as pd, warnings
from scipy import stats
from cap11_funciones import crudo_mensual, vecm
warnings.simplefilter("ignore")

m = crudo_mensual(); mes = m.index.to_numpy(); L = np.log(m[["Price_b", "Price_w"]].to_numpy()); T = len(L); k = 2
t0 = int(np.flatnonzero(mes == "2005-01")[0])               # primer mes pronosticado (índice desde 0)
dYall = np.diff(L, axis=0); pv = np.full((T, 2), np.nan); pd_ = np.full((T, 2), np.nan)
for t in range(t0, T):                                       # ventana creciente: datos hasta t-1
    v = vecm(L[:t], k); x_d = np.concatenate([[1.0], dYall[t - 2], dYall[t - 3]])   # (1, Delta y_{t-1}, Delta y_{t-2})
    pv[t] = np.concatenate([L[t - 1] @ v["beta"], x_d]) @ v["B"]
    J = v["J"]; Bd = np.linalg.lstsq(J["Z"], J["dY"], rcond=None)[0]; pd_[t] = x_d @ Bd
real = dYall[t0 - 1:]; ev = real - pv[t0:]; ed = real - pd_[t0:]; er = real
print(f"Pronósticos a un mes de Delta log precio: {mes[t0]} a {mes[-1]} ({len(real)} meses)\n")

def cw(e1, e2, p1, p2):                                      # Clark-West: modelo 1 anidado en el 2
    f = e1**2 - (e2**2 - (p1 - p2)**2); n = len(f); u = f - f.mean(); Lg = int(np.floor(4 * (n / 100)**(2 / 9)))
    s = (u**2).sum() / n
    for j in range(1, Lg + 1): s += 2 * (1 - j / (Lg + 1)) * (u[j:] * u[:-j]).sum() / n
    t = f.mean() / np.sqrt(s / n); return t, stats.norm.sf(t)
for i, nm in enumerate(["Brent", "WTI"]):
    print(f"{nm}: RECM (%): paseo aleatorio = {100 * np.sqrt((er[:, i]**2).mean()):.3f}; VAR en diferencias = "
          f"{100 * np.sqrt((ed[:, i]**2).mean()):.3f}; VECM = {100 * np.sqrt((ev[:, i]**2).mean()):.3f}")
    t, p = cw(ed[:, i], ev[:, i], pd_[t0:, i], pv[t0:, i])
    print(f"  Clark-West, VAR en diferencias frente a VECM: t = {t:.2f} (p = {p:.4f})")
    print(f"  R2 fuera de muestra del VECM frente al VAR en diferencias: {100 * (1 - (ev[:, i]**2).sum() / (ed[:, i]**2).sum()):.2f} %")

# ---- Figura: diferencia acumulada de errores cuadráticos (VAR en diferencias menos VECM) ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fe = pd.to_datetime(mes[t0:])
fig, ax = plt.subplots(figsize=(8.6, 3.3))
for i, (nm, col, ls) in enumerate([("Brent", eg.AZUL, "-"), ("WTI", eg.LADRILLO, "--")]):
    ax.plot(fe, np.cumsum(ed[:, i]**2 - ev[:, i]**2) * 1e4, color=col, ls=ls, lw=1.2, label=nm)
ax.axhline(0, color="black", lw=0.5); ax.set_ylabel("$\\sum (e^2_{\\mathrm{VAR}} - e^2_{\\mathrm{VECM}})$ (%$^2$)")
ax.legend(fontsize=8); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.savefig("graficos/cap11_ej05_pronostico.pdf")
