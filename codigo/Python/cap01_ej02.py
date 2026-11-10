# Capítulo 1, Ejercicio 1.2: rendimientos simples y logarítmicos del índice BVL (BCRP) y del S&P 500 (Yahoo)
import numpy as np, pandas as pd

bvl = pd.read_csv("datos/peru/bvl_indice_general_bcrp.csv", parse_dates=["fecha"], index_col="fecha")["indice"]
sp = pd.read_csv("datos/peru/gspc_yahoo.csv", parse_dates=["fecha"], index_col="fecha")["cierre"]

# Depuración del índice BVL: el 21-jun-2024 el BCRP registra 22 521,03 entre 29 857,57 y 30 058,66
print("BVL alrededor del dato atípico:\n", bvl["2024-06-19":"2024-06-25"].to_string(), sep="")
bvl = bvl.drop(pd.Timestamp("2024-06-21"))
print(f"Días con índice repetido (rendimiento cero): {(bvl.diff() == 0).sum()}\n")

res = {}
for nom, x in [("BVL", bvl), ("SP500", sp)]:
    R = (x / x.shift(1) - 1).dropna()            # rendimiento simple
    r = np.log(x).diff().dropna()               # rendimiento logarítmico
    dif = R - r
    print(f"{nom}: T = {len(r)}, r <= R en todas las fechas: {(r <= R + 1e-15).all()}")
    print(f"   mayor diferencia R - r = {dif.max():.5f} el {dif.idxmax().date()} "
          f"(R = {R[dif.idxmax()]:.4f}, r = {r[dif.idxmax()]:.4f})")
    for frec, nomf in [("ME", "mensual"), ("YE", "anual")]:
        suma = r.groupby(pd.Grouper(freq=frec)).sum()                         # Prop. de aditividad
        prod = np.log((1 + R).groupby(pd.Grouper(freq=frec)).prod())         # Prop. multiplicativa
        print(f"   {nomf:8s}: períodos = {len(suma)}, máx |suma log - log prod| = {(suma - prod).abs().max():.1e}")
    res[nom] = r

anual = pd.DataFrame({k: np.exp(v.groupby(v.index.year).sum()) - 1 for k, v in res.items()})
print("\nRendimientos simples anuales (%), 2026 hasta noviembre:")
print((100 * anual).round(2).to_string())
desv = pd.DataFrame({k: [v.mean(), v.std(), v.min(), v.max()] for k, v in res.items()},
                    index=["media", "desv_est", "mínimo", "máximo"])
print("\nRendimientos logarítmicos diarios:\n", desv.round(5).to_string(), sep="")

# ---- Figura: rendimientos logarítmicos diarios de ambos índices ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
for a, (nom, col, tit) in zip(ax, [("BVL", eg.AZUL, "S&P/BVL Perú General (BCRP, soles)"),
                                   ("SP500", eg.LADRILLO, "S&P 500 (Yahoo Finance, dólares)")]):
    a.plot(res[nom].index, res[nom], color=col, lw=0.5)
    a.set_title(tit, loc="left"); a.set_ylim(-0.14, 0.11); a.set_ylabel("$r_t$"); eg.coma(a)
import matplotlib.dates as mdates
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.savefig("graficos/cap01_fig_logsimple.pdf")
