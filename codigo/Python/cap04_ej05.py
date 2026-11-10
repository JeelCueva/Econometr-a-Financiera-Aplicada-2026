# Capítulo 4, Ejercicio 4.5: ¿cambió la beta de Utils? Chow, sup-F y CUSUM
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv")
y = (d["Utils"] - d["RF"]).to_numpy(); X = np.column_stack([np.ones(len(y)), d["MktRF"]]); T, k = len(y), 2
def mco(i):
    b = np.linalg.lstsq(X[i], y[i], rcond=None)[0]; e = y[i] - X[i] @ b; return b, e @ e
def chow(tb):                               # tb = índice (base 0) de la primera obs. del segundo régimen
    S = mco(slice(0, T))[1]; S1 = mco(slice(0, tb))[1]; S2 = mco(slice(tb, T))[1]
    return ((S - S1 - S2) / k) / ((S1 + S2) / (T - 2 * k))

tb = int(np.where(d.dates == "1992-10")[0][0]); F = chow(tb)
print(f"Chow con quiebre en 1992-10: F = {F:.2f}  p = {stats.f.sf(F, k, T - 2*k):.2e}")
b1, b2 = mco(slice(0, tb))[0], mco(slice(tb, T))[0]
print(f"Beta antes = {b1[1]:.3f}  después = {b2[1]:.3f} | alfa anual antes = {1200*b1[0]:.2f} %  después = {1200*b2[0]:.2f} %")

Dq = (np.arange(T) >= tb).astype(float)
Z = np.column_stack([X, Dq, Dq * X[:, 1]]); ZtZi = np.linalg.inv(Z.T @ Z)
bz = ZtZi @ Z.T @ y; ez = y - Z @ bz
V = ZtZi @ ((Z * ez[:, None]).T @ (Z * ez[:, None])) @ ZtZi * T / (T - 4)
W = bz[2:] @ np.linalg.solve(V[2:, 2:], bz[2:])
print(f"Wald robusto (HC1): W = {W:.2f}  p = {stats.chi2.sf(W, 2):.2e}")

rango = np.arange(int(np.floor(0.15 * T)) - 1, int(np.ceil(0.85 * T)))   # base 0: mismas fechas que en R
Fs = np.array([chow(t) for t in rango])
print(f"sup-F = {Fs.max():.2f} en {d.dates[rango[Fs.argmax()]]} (valor crítico asintótico al 5 %: 11,7 / 2 = 5,85)")

w = []
for t in range(k, T):
    Xa = X[:t]; A = np.linalg.inv(Xa.T @ Xa); ba = A @ Xa.T @ y[:t]
    w.append((y[t] - X[t] @ ba) / np.sqrt(1 + X[t] @ A @ X[t]))
w = np.array(w); W_t = np.cumsum(w) / w.std(ddof=1)
banda = 0.948 * (np.sqrt(T - k) + 2 * np.arange(1, len(w) + 1) / np.sqrt(T - k))
cruce = np.where(np.abs(W_t) > banda)[0]   # ojo: argmax devolvería 0 si no hay cruces
print(f"CUSUM: máximo |W_t| / banda = {np.max(np.abs(W_t) / banda):.2f}; "
      + ("no cruza las bandas" if len(cruce) == 0 else f"cruza en {d.dates[k + cruce[0]]}"))

# ---- Figura: F de Chow para cada fecha y CUSUM ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
f_F = pd.to_datetime(d.dates[rango]); ax[0].plot(f_F, Fs, color=eg.AZUL, lw=1)
ax[0].axhline(5.85, color=eg.LADRILLO, ls="--", lw=1, label="valor crítico sup-F al 5 %")
ax[0].axvline(pd.Timestamp("1992-10-01"), color=eg.GRIS, ls=":", lw=1, label="octubre de 1992")
ax[0].set_title("Estadístico F de Chow según la fecha de quiebre"); ax[0].legend(fontsize=8)
f_C = pd.to_datetime(d.dates[k:]); ax[1].plot(f_C, W_t, color=eg.AZUL, lw=1, label="CUSUM $W_t$")
ax[1].plot(f_C, banda, color=eg.LADRILLO, ls="--", lw=1, label="bandas al 5 %"); ax[1].plot(f_C, -banda, color=eg.LADRILLO, ls="--", lw=1)
ax[1].set_title("CUSUM de residuos recursivos"); ax[1].legend(fontsize=8, loc="lower left")
for a in ax: a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.savefig("graficos/cap04_fig_quiebres.pdf")
