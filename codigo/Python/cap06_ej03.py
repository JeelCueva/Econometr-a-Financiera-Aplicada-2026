# Capítulo 6, Ejercicio 6.3: ¿pronostica el ARMA(1,1) el VIX mejor que un paseo aleatorio? (fuera de muestra)
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA

v = pd.read_csv("datos/internacional/vix_diario.csv"); y = np.log(v["CLOSE"].to_numpy()); T = len(y)
fin = int(np.where(v.DATE <= "2015-12-31")[0].max())       # último día de estimación (índice desde 0)
m = ARIMA(y[:fin + 1], order=(1, 0, 1), trend="c").fit()
mu, phi, th = m.params[:3]
print(f"Estimación {v.DATE[0]} a {v.DATE[fin]}: mu = {mu:.4f}  phi = {phi:.4f}  theta = {th:.4f}")
e = np.zeros(T)
for t in range(1, T): e[t] = (y[t] - mu) - phi * (y[t - 1] - mu) - th * e[t - 1]
media_est = y[:fin + 1].mean()
def dm(d, h):                                              # Diebold-Mariano, Newey-West con L = h - 1
    n = len(d); u = d - d.mean(); S = u @ u / n
    for j in range(1, h): S += 2 * (1 - j / h) * (u[j:] @ u[:-j]) / n
    return d.mean() / np.sqrt(S / n)
print(f"Pronósticos desde {v.DATE[fin]}; {T - 1 - fin} orígenes\n")
print("   h   RMSE ARMA   RMSE paseo   RMSE media   ARMA/paseo   DM (paseo - ARMA)")
guardar = {}
for h in (1, 5, 22):
    o = np.arange(fin, T - h)
    f1 = mu + phi * (y[o] - mu) + th * e[o]; fa = mu + phi**(h - 1) * (f1 - mu)
    ea, ep, em = y[o + h] - fa, y[o + h] - y[o], y[o + h] - media_est
    r = np.sqrt([np.mean(ea**2), np.mean(ep**2), np.mean(em**2)])
    print(f"{h:4d} {r[0]:11.4f} {r[1]:12.4f} {r[2]:12.4f} {r[0]/r[1]:12.3f} {dm(ep**2 - ea**2, h):16.2f}")
    guardar[h] = (o, fa)

# ---- Figura: pronósticos a 22 días frente a lo observado, 2019-2021 ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
o, fa = guardar[22]; fechas = pd.to_datetime(v.DATE)
sel = (fechas[o + 22] >= "2019-07-01") & (fechas[o + 22] <= "2021-06-30")
fig, ax = plt.subplots(figsize=(8.4, 3.4))
ax.plot(fechas[o + 22][sel], np.exp(y[o + 22][sel]), color="black", lw=0.8, label="VIX observado")
ax.plot(fechas[o + 22][sel], np.exp(fa[sel]), color=eg.LADRILLO, lw=1, label="ARMA(1,1), pronóstico hecho 22 días antes")
ax.plot(fechas[o + 22][sel], np.exp(y[o][sel]), color=eg.AZUL, lw=1, ls="--", label="paseo aleatorio (valor de hace 22 días)")
ax.set_ylabel("VIX"); ax.legend(fontsize=8)
fig.savefig("graficos/cap06_fig_vix_pronostico.pdf")
