# Capítulo 7, Ejercicio 7.4: prueba de rachas (Wald-Wolfowitz) sobre los signos
import numpy as np, pandas as pd
from cap07_funciones import rachas
from statsmodels.sandbox.stats.runs import runstest_1samp     # solo para comprobar

b = pd.read_csv("datos/internacional/brent_diario.csv"); r = np.diff(np.log(b["Price"].to_numpy()))
f = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv")
series = {"Brent diario": r[r != 0], "S1V3 mensual": f["S1V3"].to_numpy()[f["S1V3"] != 0],
          "S5V3 mensual": f["S5V3"].to_numpy()[f["S5V3"] != 0]}
print(f"Brent: {(r == 0).sum()} días con rendimiento exactamente cero excluidos")
print("Serie            n1(+)  n2(-)  rachas  esperadas       z | statsmodels z")
for nm, x in series.items():
    k = rachas(x > 0); zs, _ = runstest_1samp((x > 0).astype(float), cutoff=0.5, correction=False)
    print(f"{nm:<14} {k['n1']:6d} {k['n2']:6d} {k['R']:7d} {k['ER']:10.1f} {k['z']:7.2f} | {zs:7.2f}")

# ---- Figura: longitudes de racha observadas frente a las esperadas bajo independencia ----
import estilo_graficos as eg, matplotlib.pyplot as plt
def longitudes(x):
    cortes = np.flatnonzero(x[1:] != x[:-1]) + 1
    return np.diff(np.concatenate([[0], cortes, [len(x)]]))
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.3))
for a, nm in zip(ax, ["Brent diario", "S1V3 mensual"]):
    x = series[nm] > 0; L = longitudes(x); p = x.mean(); kmax = 8
    obs = np.array([(L == k).mean() for k in range(1, kmax + 1)])
    # bajo independencia, la longitud de una racha es geométrica; se promedia entre rachas de + y de -
    teo = np.array([0.5 * ((1 - p) * p**(k - 1) + p * (1 - p)**(k - 1)) for k in range(1, kmax + 1)])
    kk = np.arange(1, kmax + 1); w = 0.38
    a.bar(kk - w / 2, obs, w, color=eg.LADRILLO, label="observada")
    a.bar(kk + w / 2, teo, w, color=eg.AZUL, label="esperada (independencia)")
    a.set_title(nm); a.set_xlabel("longitud de la racha"); a.legend(fontsize=8); eg.coma(a)
ax[0].set_ylabel("proporción de rachas")
fig.savefig("graficos/cap07_ej04_rachas.pdf")
