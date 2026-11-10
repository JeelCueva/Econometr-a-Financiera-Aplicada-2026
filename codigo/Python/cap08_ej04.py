# Capítulo 8, Ejercicio 8.4: pronóstico de la volatilidad del Brent con el GARCH(1,1)-t
import numpy as np, pandas as pd
from cap08_funciones import garch_fit
from arch import arch_model                        # solo para comprobar

b = pd.read_csv("datos/internacional/brent_diario.csv"); r = 100 * np.diff(np.log(b["Price"].to_numpy())); T = len(r)
m = garch_fit(r, "t"); p = m["par"]
pers = p["alpha"] + p["beta"]; s2bar = p["omega"] / (1 - pers)
eT = r[-1] - p["mu"]; s2T = m["s2"][-1]
s2T1 = p["omega"] + p["alpha"] * eT**2 + p["beta"] * s2T              # varianza de mañana
print(f"último dato: {b['Date'].iloc[-1]}; r_T = {r[-1]:.3f} %; sigma2_T = {s2T:.4f}")
print(f"sigma2_T+1 = {p['omega']:.4f} + {p['alpha']:.4f} x {eT**2:.4f} + {p['beta']:.4f} x {s2T:.4f} = {s2T1:.4f}")
print(f"varianza incondicional = {s2bar:.3f} (volatilidad anual {np.sqrt(252 * s2bar):.1f} %)")
h = np.arange(1, 253)
s2h = s2bar + pers**(h - 1) * (s2T1 - s2bar)                           # E_T[sigma2_{T+h}]
for k in [1, 5, 21, 63, 126, 252]:
    print(f"h = {k:3d}: E[sigma2] = {s2h[k - 1]:.3f}  (vol. diaria {np.sqrt(s2h[k - 1]):.3f} %; "
          f"anualizada {np.sqrt(252 * s2h[k - 1]):.1f} %)")
for H in [21, 252]:
    v = s2h[:H].sum()
    print(f"varianza acumulada a {H:3d} días = {v:6.0f} -> volatilidad del período {np.sqrt(v):.2f} % "
          f"(regla de la raíz: {np.sqrt(H * s2T1):.2f} %)")
a = arch_model(r, dist="t").fit(disp="off"); fc = a.forecast(horizon=252).variance.iloc[-1].to_numpy()
print(f"\ncomprobación con arch: h = 1: {fc[0]:.3f}; h = 21: {fc[20]:.3f}; h = 252: {fc[251]:.3f}")

# ---- Figura: volatilidad de los dos últimos años y su pronóstico ----
import estilo_graficos as eg, matplotlib.pyplot as plt
n = 504; fe = pd.to_datetime(b["Date"].iloc[1:]).to_numpy()[-n:]
fut = pd.bdate_range(pd.Timestamp(b["Date"].iloc[-1]) + pd.Timedelta(days=1), periods=252)
fig, ax = plt.subplots(figsize=(9.0, 3.5))
ax.plot(fe, np.sqrt(252 * m["s2"][-n:]), color=eg.AZUL, lw=0.9, label="volatilidad condicional estimada")
ax.plot(fut, np.sqrt(252 * s2h), color=eg.LADRILLO, lw=1.3, ls="--", label="pronóstico $\\mathrm{E}_T[\\sigma^2_{T+h}]$")
ax.axhline(np.sqrt(252 * s2bar), color="black", ls=":", lw=0.8, label="nivel incondicional")
ax.set_ylabel("volatilidad anualizada (%)"); ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.savefig("graficos/cap08_ej04_pronostico.pdf")
