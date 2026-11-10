# Capítulo 7, Ejercicio 7.5: el factor momentum como anomalía de la eficiencia débil
import numpy as np, pandas as pd, statsmodels.api as sm

f = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv")
m = f["Mom"].to_numpy(); T = len(m)
L = int(np.floor(4 * (T / 100)**(2 / 9)))                 # regla de Newey-West
nw = sm.OLS(m, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": L})
e = m - m.mean()
print(f"T = {T}, media mensual = {m.mean():.4f} (anual {1200 * m.mean():.2f} %), "
      f"desv. est. anual = {100 * m.std(ddof=1) * np.sqrt(12):.2f} %")
print(f"t clásico = {m.mean() / (m.std(ddof=1) / np.sqrt(T)):.2f}; t Newey-West (L = {L}) = {nw.tvalues[0]:.2f}")
print(f"razón de Sharpe anual = {np.sqrt(12) * m.mean() / m.std(ddof=1):.3f}; "
      f"asimetría = {(e**3).mean() / (e**2).mean()**1.5:.3f}")
w = np.cumsum(np.log1p(m)); dd = w - np.maximum.accumulate(w)
print(f"peor mes: {f['dates'][m.argmin()]} ({100 * m.min():.2f} %); caída máxima acumulada = "
      f"{100 * (np.exp(dd.min()) - 1):.1f} % (fondo en {f['dates'][dd.argmin()]})")
for n, sel in {"1949-1999": f["dates"] <= "1999-12", "2000-2017": f["dates"] >= "2000-01"}.items():
    x = m[sel.to_numpy()]
    print(f"  {n}: media anual = {1200 * x.mean():6.2f} %, t = {x.mean() / (x.std(ddof=1) / np.sqrt(len(x))):5.2f}")

# ---- Figura: riqueza acumulada y caídas del factor momentum frente al mercado ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fechas = pd.to_datetime(f["dates"]); wm = np.cumsum(np.log1p(f["MktRF"].to_numpy()))
fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
ax[0].plot(fechas, w, color=eg.LADRILLO, lw=1, label="momentum (UMD)")
ax[0].plot(fechas, wm, color=eg.AZUL, lw=1, ls="--", label="mercado menos tasa libre")
ax[0].set_ylabel("log riqueza acumulada"); ax[0].legend(fontsize=8)
ax[1].fill_between(fechas, 100 * (np.exp(dd) - 1), 0, color=eg.LADRILLO, alpha=0.5)
ax[1].set_ylabel("caída desde\nel máximo (%)")
for a in ax: a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.savefig("graficos/cap07_ej05_momentum.pdf")
