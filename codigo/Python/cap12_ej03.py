# Capítulo 12, Ejercicio 12.3: DCC trivariado (S&P 500, Nasdaq y Brent) y correlaciones en la crisis de 2008
import numpy as np, pandas as pd
from cap12_funciones import panel_diario, etapa1, dcc_fit

f, R = panel_diario(); Z, S2, _ = etapa1(R); m = dcc_fit(Z)
print(f"DCC(1,1) trivariado: a = {m['a']:.4f} (ee {m['ee'][0]:.4f}), b = {m['b']:.4f} (ee {m['ee'][1]:.4f}); a + b = {m['a'] + m['b']:.4f}")
print(f"LR frente a CCC = {2 * (m['ll'] - m['ll_ccc']):.2f} (2 g.l.)\n")
per = {"calma 2006-01 a 2008-08": (f >= "2006-01-01") & (f <= "2008-08-31"),
       "crisis 2008-09 a 2009-03": (f >= "2008-09-01") & (f <= "2009-03-31"), "2017-2018": f >= "2017-01-01"}
pares = {"S&P-Nasdaq": (0, 1), "S&P-Brent": (0, 2), "Nasdaq-Brent": (1, 2)}
print(f"{'promedio de la correlación DCC':<26} " + " ".join(f"{p:>12}" for p in pares))
for nm, k in per.items(): print(f"{nm:<26} " + " ".join(f"{m['R'][k, i, j].mean():12.3f}" for i, j in pares.values()))
k = per["crisis 2008-09 a 2009-03"]
print(f"\nvolatilidad anual promedio (GARCH) en la crisis: S&P {np.sqrt(252 * S2[k, 0]).mean():.1f} %, "
      f"Nasdaq {np.sqrt(252 * S2[k, 1]).mean():.1f} %, Brent {np.sqrt(252 * S2[k, 2]).mean():.1f} %")
print(f"menor autovalor de R_t en toda la muestra: {np.linalg.eigvalsh(m['R']).min():.4f} (todas las R_t son definidas positivas)")

# ---- Figura: las tres correlaciones DCC ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fe = pd.to_datetime(f)
fig, ax = plt.subplots(figsize=(9.0, 3.6))
for (nm, (i, j)), col, ls in zip(pares.items(), [eg.AZUL, eg.LADRILLO, eg.OLIVA], ["-", "-", "--"]):
    ax.plot(fe, m["R"][:, i, j], color=col, ls=ls, lw=0.9, label=nm)
ax.axvspan(pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-31"), color=eg.GRIS, alpha=0.15)
ax.axhline(0, color="black", lw=0.5); ax.set_ylabel("correlación condicional"); ax.legend(fontsize=8, ncol=3, loc="lower right")
eg.coma(ax); ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
fig.savefig("graficos/cap12_ej03_dcc3.pdf")
