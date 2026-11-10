# Capítulo 8, Ejercicio 8.2: GARCH(1,1) normal para el Brent, estimado a mano por máxima verosimilitud
import numpy as np, pandas as pd
from cap08_funciones import garch_fit
from arch import arch_model                        # solo para comprobar

b = pd.read_csv("datos/internacional/brent_diario.csv"); r = 100 * np.diff(np.log(b["Price"].to_numpy())); T = len(r)
m = garch_fit(r, "normal"); p = m["par"]
print(f"T = {T}; log-verosimilitud = {m['ll']:.4f}; varianza inicial (backcast) = {m['bc']:.4f}")
tab = pd.DataFrame({"coef": p, "ee_hess": m["ee"], "ee_rob": m["ee_rob"]})
tab["t_hess"] = (tab["coef"] / tab["ee_hess"]).round(2); tab["t_rob"] = (tab["coef"] / tab["ee_rob"]).round(2)
print(tab[["coef", "ee_hess", "t_hess", "ee_rob", "t_rob"]].round(4))
pers = p["alpha"] + p["beta"]; s2bar = p["omega"] / (1 - pers)
print(f"persistencia alpha + beta = {pers:.5f}; vida media = {np.log(0.5) / np.log(pers):.1f} días")
print(f"varianza incondicional = {s2bar:.3f} (muestral {r.var(ddof=1):.3f}); volatilidad anual = {np.sqrt(252 * s2bar):.1f} %")
print(f"condición de cuarto momento: (a+b)^2 + 2a^2 = {pers**2 + 2 * p['alpha']**2:.4f}")
vol = np.sqrt(252 * m["s2"]); f = b["Date"].iloc[1:].to_numpy()
v1, f1 = vol[250:], f[250:]                         # se omite el primer año (arranque)
print(f"volatilidad anualizada desde {f1[0]}: mínima {v1.min():.1f} % ({f1[v1.argmin()]}), máxima {v1.max():.1f} % ({f1[v1.argmax()]})")
a = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
print("\ncomprobación con arch (ee robustos):")
print(pd.DataFrame({"coef": a.params, "ee_rob": a.std_err}).round(4), f"\nlog-verosimilitud = {a.loglikelihood:.4f}")

# ---- Figura: rendimientos con bandas de +-2 sigma_t y volatilidad anualizada ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fe = pd.to_datetime(f); sd = np.sqrt(m["s2"])
fig, ax = plt.subplots(2, 1, figsize=(9.4, 5.0), sharex=True)
ax[0].plot(fe, r, color=eg.GRIS, lw=0.3, alpha=0.6, label="rendimiento diario (%)")
ax[0].plot(fe, 2 * sd, color=eg.LADRILLO, lw=1.0, label="$\\pm2\\,\\hat\\sigma_t$"); ax[0].plot(fe, -2 * sd, color=eg.LADRILLO, lw=1.0)
ax[0].set_ylim(-25, 25); ax[0].legend(fontsize=8, loc="lower left")
ax[1].plot(fe, vol, color=eg.AZUL, lw=0.8); ax[1].axhline(np.sqrt(252 * s2bar), color="black", ls=":", lw=0.8)
ax[1].set_ylabel("volatilidad anualizada (%)")
for x in ax: x.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.savefig("graficos/cap08_ej02_volatilidad.pdf")
