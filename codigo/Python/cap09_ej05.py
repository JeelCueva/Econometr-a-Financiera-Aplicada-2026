# Capítulo 9, Ejercicio 9.5: GARCH en media: ¿pagan más las acciones cuando la volatilidad es mayor?
import numpy as np, pandas as pd
from cap09_funciones import ajustar

f = pd.read_csv("datos/internacional/french_mercado_mensual_1926_2018.csv"); x = f["Mkt-RF"].to_numpy(); T = len(x)
print(f"Exceso de rendimiento mensual del mercado (%), {f['dates'].iloc[0]} a {f['dates'].iloc[-1]}, T = {T}")
print(f"media = {x.mean():.4f}; varianza = {x.var(ddof=1):.4f}\n")
casos = {"GARCH-M normal": ("garch", "normal"), "GARCH-M t": ("garch", "t"), "GJR-M t": ("gjr", "t")}
res = {}
for nm, (mod, d) in casos.items():
    m = ajustar(x, mod, "m", d); p = dict(zip(m["nombres"], m["p"])); ee = dict(zip(m["nombres"], m["ee_rob"]))
    pr = p["lambda"] * m["s2"].mean(); res[nm] = (m, p)
    print(f"{nm:<15} lambda = {p['lambda']:.5f} (ee rob. {ee['lambda']:.5f}, t = {p['lambda'] / ee['lambda']:.2f}); "
          f"mu = {p['mu']:.4f}; log-ver = {m['ll']:.2f}")
    print(f"{'':<15} prima por volatilidad promedio: lambda x media(sigma2) = {pr:.4f} % mensual ({12 * pr:.2f} % anual)")

# ---- Figura: volatilidad condicional y prima esperada implícita (GARCH-M t) ----
import estilo_graficos as eg, matplotlib.pyplot as plt
m, p = res["GARCH-M t"]; fe = pd.to_datetime(f["dates"])
fig, ax = plt.subplots(2, 1, figsize=(9.0, 4.6), sharex=True)
ax[0].plot(fe, np.sqrt(12 * m["s2"]), color=eg.AZUL, lw=0.8); ax[0].set_ylabel("volatilidad\nanualizada (%)")
ax[1].plot(fe, 12 * (p["mu"] + p["lambda"] * m["s2"]), color=eg.LADRILLO, lw=0.8)
ax[1].axhline(12 * x.mean(), color="black", ls=":", lw=0.8)
ax[1].set_ylabel("exceso esperado\nanualizado (%)")
for a in ax: a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.savefig("graficos/cap09_ej05_garch_m.pdf")
