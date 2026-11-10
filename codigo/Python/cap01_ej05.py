# Capítulo 1, Ejercicio 1.5: media aritmética y geométrica, anualización y días de negociación por año
import numpy as np, pandas as pd

lee = lambda f, c: pd.read_csv(f"datos/peru/{f}", parse_dates=["fecha"], index_col="fecha")[c]
P = {"BVN": lee("bvn_yahoo.csv", "cierre_ajustado"), "EPU": lee("epu_yahoo.csv", "cierre_ajustado"),
     "SP500": lee("gspc_yahoo.csv", "cierre")}
bvl = lee("bvl_indice_general_bcrp.csv", "indice").drop(pd.Timestamp("2024-06-21"))

filas, vol = {}, {}
for k, x in P.items():
    x = x[:"2025-12-31"]                                       # años completos 2015-2025
    Ra = x.groupby(x.index.year).last().pct_change().dropna()  # rendimientos simples anuales
    T = len(Ra); A = Ra.mean(); s2 = ((Ra - A) ** 2).mean()
    G = np.prod(1 + Ra) ** (1 / T) - 1
    filas[k] = {"T": T, "A_%": 100*A, "G_%": 100*G, "A-s2/2_%": 100*(A - s2/2), "A-G_pb": 1e4*(A-G),
                "s2/2_pb": 1e4*s2/2}
    rd = np.log(x).diff().dropna(); ra = np.log1p(Ra)
    vol[k] = {"sd_diaria_%": 100*rd.std(), "x raiz(252)_%": 100*rd.std()*np.sqrt(252),
              "x raiz(dias)_%": 100*rd.std()*np.sqrt(rd.groupby(rd.index.year).size().mean()),
              "sd_anual_obs_%": 100*ra.std()}
print("(a) Medias de los rendimientos simples anuales, 2015-2025:")
print(pd.DataFrame(filas).T.round(2).to_string())
print("\n(b) Volatilidad anualizada (rendimientos logarítmicos):")
print(pd.DataFrame(vol).T.round(2).to_string())

dias = pd.DataFrame({"NYSE (Yahoo)": P["SP500"].groupby(P["SP500"].index.year).size(),
                     "BVL (BCRP)": bvl.groupby(bvl.index.year).size()}).loc[2015:2025]
print("\n(c) Días con dato por año:")
print(dias.T.to_string())
print("Promedio:", dias.mean().round(1).to_dict())

# ---- Figura: (a) A, G y A - s^2/2; (b) volatilidad diaria anualizada frente a la observada ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
F, V = pd.DataFrame(filas).T, pd.DataFrame(vol).T
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
x = np.arange(3)
for j, (c, col, lab) in enumerate([("A_%", eg.AZUL, "aritmética $A$"), ("G_%", eg.LADRILLO, "geométrica $G$"),
                                    ("A-s2/2_%", eg.OLIVA, "$A-s^2/2$")]):
    ax[0].bar(x + (j - 1) * 0.26, F[c], width=0.26, color=col, label=lab)
ax[0].set_xticks(x); ax[0].set_xticklabels(F.index); ax[0].set_ylabel("% anual"); ax[0].legend(fontsize=8)
for j, (c, col, lab) in enumerate([("x raiz(252)_%", eg.AZUL, "$\\hat\\sigma_d\\sqrt{252}$"),
                                    ("sd_anual_obs_%", eg.LADRILLO, "desv. est. de $r$ anuales")]):
    ax[1].bar(x + (j - 0.5) * 0.35, V[c], width=0.35, color=col, label=lab)
ax[1].set_xticks(x); ax[1].set_xticklabels(V.index); ax[1].set_ylabel("% anual"); ax[1].legend(fontsize=8)
for a in ax: eg.coma(a); a.set_xticks(x); a.set_xticklabels(F.index)
fig.savefig("graficos/cap01_fig_anualizar.pdf")
