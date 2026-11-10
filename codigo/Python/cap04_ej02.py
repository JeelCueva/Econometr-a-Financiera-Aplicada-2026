# Capítulo 4, Ejercicio 4.2: alfas y betas de las 12 industrias con errores robustos
import numpy as np, pandas as pd
import statsmodels.api as sm

d = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv")
industrias = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
              "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]
T = len(d); X = sm.add_constant(d["MktRF"].to_numpy())
L = int(np.floor(4 * (T / 100)**(2 / 9)))     # regla de Newey y West (1994)
print(f"Rezagos de Newey-West: L = {L}\n")
filas = {}
for i in industrias:
    y = (d[i] - d["RF"]).to_numpy()
    cl = sm.OLS(y, X).fit()
    hc = sm.OLS(y, X).fit(cov_type="HC1")
    nw = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": False})
    filas[i] = {"t_alfa_cl": cl.tvalues[0], "t_alfa_HC1": hc.tvalues[0], "t_alfa_NW": nw.tvalues[0],
                "ee_beta_cl": cl.bse[1], "ee_beta_HC1": hc.bse[1], "ee_beta_NW": nw.bse[1]}
tabla = pd.DataFrame(filas).T
print(tabla.round(4).to_string())
sig = (tabla.iloc[:, :3].abs() > 1.96).sum()
print(f"\nAlfas significativas al 5 %: clásico {sig.iloc[0]} | HC1 {sig.iloc[1]} | Newey-West {sig.iloc[2]}")
print(f"Cociente medio ee(beta) NW/clásico: {(tabla.ee_beta_NW / tabla.ee_beta_cl).mean():.3f}")

# ---- Figura: errores estándar de beta y estadísticos t de alfa ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
pos = np.arange(len(tabla)); w = 0.27
for j, (c, col, nom) in enumerate([("ee_beta_cl", eg.GRIS, "clásico"), ("ee_beta_HC1", eg.AZUL, "HC1"),
                                   ("ee_beta_NW", eg.LADRILLO, "Newey-West")]):
    ax[0].bar(pos + (j - 1) * w, tabla[c], w, color=col, label=nom)
    ax[1].bar(pos + (j - 1) * w, tabla[c.replace("ee_beta", "t_alfa")], w, color=col, label=nom)
for a in ax:
    a.set_xticks(pos); a.set_xticklabels(tabla.index, rotation=60, fontsize=8); eg.coma(a)
    a.xaxis.set_major_formatter(plt.FixedFormatter(tabla.index))
ax[0].set_title(r"Error estándar de $\hat\beta$"); ax[0].legend(fontsize=8)
for s in (1.96, -1.96): ax[1].axhline(s, color="black", ls=":", lw=0.8)
ax[1].set_title(r"Estadístico $t$ de $H_0:\alpha=0$")
fig.savefig("graficos/cap04_fig_robustos.pdf")
