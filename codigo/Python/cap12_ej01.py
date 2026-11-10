# Capítulo 12, Ejercicio 12.1: correlaciones que cambian: ventana móvil frente a EWMA
import numpy as np, pandas as pd
from cap12_funciones import panel_diario, ewma_cov

f, R = panel_diario(); T = len(R)
print(f"Panel diario: {f[0]} a {f[-1]}, T = {T} días comunes al S&P 500, al Nasdaq y al Brent")
print("Correlaciones de toda la muestra:"); print(np.round(np.corrcoef(R.T), 4))
H = ewma_cov(R); rho_e = lambda i, j: H[:, i, j] / np.sqrt(H[:, i, i] * H[:, j, j])
def mov(i, j, n=250):
    return np.concatenate([np.full(n, np.nan), [np.corrcoef(R[t - n:t, i], R[t - n:t, j])[0, 1] for t in range(n, T)]])
res = {}
for nm, (i, j) in {"S&P-Brent": (0, 2), "S&P-Nasdaq": (0, 1)}.items():
    e, m = rho_e(i, j), mov(i, j); res[nm] = (e, m); e2, f2 = e[30:], f[30:]
    print(f"\n{nm}: EWMA mín. {e2.min():.3f} ({f2[e2.argmin()]}), máx. {e2.max():.3f} ({f2[e2.argmax()]}); "
          f"ventana de 250 días: mín. {np.nanmin(m):.3f}, máx. {np.nanmax(m):.3f}")
    for a, b in [("1999", "2007"), ("2009", "2018")]:
        k = (pd.Series(f).str[:4] >= a).to_numpy() & (pd.Series(f).str[:4] <= b).to_numpy()
        print(f"  {a}-{b}: correlación muestral = {np.corrcoef(R[k, i], R[k, j])[0, 1]:.3f}")

# ---- Figura: correlación EWMA y de ventana móvil ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fe = pd.to_datetime(f)
fig, ax = plt.subplots(2, 1, figsize=(9.0, 5.0), sharex=True)
for a, nm in zip(ax, res):
    e, m = res[nm]
    a.plot(fe[30:], e[30:], color=eg.GRIS, lw=0.5, label="EWMA ($\\lambda=0{,}94$)")
    a.plot(fe, m, color=eg.LADRILLO, lw=1.3, label="ventana móvil de 250 días")
    a.axhline(0, color="black", lw=0.5); a.set_title(nm, fontsize=9); a.set_ylabel("correlación"); eg.coma(a)
ax[0].legend(fontsize=8, loc="upper left"); ax[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
fig.savefig("graficos/cap12_ej01_correlaciones.pdf")
