# Capítulo 9, Ejercicio 9.2: GJR-GARCH-t para el S&P 500 diario y comparación con el VIX
import numpy as np, pandas as pd
from scipy import stats
from cap09_funciones import ajustar
from arch import arch_model                              # solo para comprobar

s = pd.read_csv("datos/internacional/sp500_diario_yahoo.csv"); r = 100 * np.diff(np.log(s["Adj Close"].to_numpy())); T = len(r)
g = ajustar(r, "garch", "c", "t"); j = ajustar(r, "gjr", "c", "t"); p = dict(zip(j["nombres"], j["p"]))
print(f"T = {T} ({s['Date'].iloc[1]} a {s['Date'].iloc[-1]})\n\nGJR-GARCH(1,1)-t:")
tab = pd.DataFrame({"coef": j["p"], "ee_rob": j["ee_rob"]}, index=j["nombres"]); tab["t_rob"] = (tab.coef / tab.ee_rob).round(2)
print(tab.round(4))
pg = dict(zip(g["nombres"], g["p"]))
print(f"\nGARCH(1,1)-t: alpha = {pg['alpha']:.4f}, beta = {pg['beta']:.4f}, log-ver = {g['ll']:.2f}")
LR = 2 * (j["ll"] - g["ll"])
print(f"GJR-GARCH-t: log-ver = {j['ll']:.2f}; LR (gamma = 0) = {LR:.2f}, p = {stats.chi2.sf(LR, 1):.2e}")
pers = p["alpha"] + p["gamma"] / 2 + p["beta"]; s2bar = p["omega"] / (1 - pers)
print(f"persistencia alpha + gamma/2 + beta = {pers:.5f}; vida media = {np.log(0.5) / np.log(pers):.1f} días")
print(f"varianza incondicional = {s2bar:.4f} (muestral {r.var(ddof=1):.4f}); volatilidad anual = {np.sqrt(252 * s2bar):.1f} %")
print(f"coeficiente de e_{{t-1}}^2: si e < 0, alpha + gamma = {p['alpha'] + p['gamma']:.4f}; si e > 0, alpha = {p['alpha']:.4f}")
v = pd.read_csv("datos/internacional/vix_diario.csv")
d = pd.DataFrame({"DATE": s["Date"].iloc[1:].to_numpy(), "vol": np.sqrt(252 * j["s2"])}).merge(v[["DATE", "CLOSE"]], on="DATE")
print(f"\nVIX: {len(d)} días en común; correlación con la volatilidad GJR = {d.vol.corr(d.CLOSE):.3f}")
print(f"promedios: VIX = {d.CLOSE.mean():.2f}; volatilidad GJR = {d.vol.mean():.2f}; diferencia = {(d.CLOSE - d.vol).mean():.2f} puntos")
a = arch_model(r, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
print(f"\ncomprobación con arch: {np.round(a.params.to_numpy(), 4)}, log-ver = {a.loglikelihood:.2f}")

# ---- Figura: volatilidad GJR anualizada frente al VIX ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fe = pd.to_datetime(d.DATE)
fig, ax = plt.subplots(figsize=(9.0, 3.5))
ax.plot(fe, d.CLOSE, color=eg.GRIS, lw=0.8, label="VIX (volatilidad implícita a 30 días)")
ax.plot(fe, d.vol, color=eg.LADRILLO, lw=0.8, label="GJR-GARCH-$t$ (volatilidad condicional anualizada)")
ax.set_ylabel("puntos porcentuales"); ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.savefig("graficos/cap09_ej02_gjr_vix.pdf")
