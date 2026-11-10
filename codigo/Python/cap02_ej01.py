# Capítulo 2, Ejercicio 2.1: momentos y normalidad de los rendimientos diarios del Brent
import numpy as np, pandas as pd
from scipy import stats

datos = pd.read_csv("datos/internacional/brent_diario.csv", parse_dates=["Date"])
r = np.log(datos["Price"]).diff().dropna()        # rendimientos logarítmicos diarios
r.index = datos["Date"].iloc[1:]

def momentos(x):
    s = stats.skew(x)                             # divisor T (bias=True por defecto)
    k = stats.kurtosis(x, fisher=False)           # curtosis (no exceso)
    jb = stats.jarque_bera(x)
    return pd.Series({"T": len(x), "media": x.mean(), "desv_est": x.std(ddof=0),
                      "asimetria": s, "curtosis": k, "JB": jb.statistic, "p_valor": jb.pvalue})

print("Período:", r.index.min().date(), "a", r.index.max().date(), "\n")
print(momentos(r).apply(lambda v: f"{v:.5g}").to_string())

print("\nLos cinco rendimientos más grandes en valor absoluto:")
print(r.loc[r.abs().nlargest(5).index].round(4).to_string())

print("\nSin el 21 y el 22 de abril de 2020:")
sin_abr20 = r.drop(pd.to_datetime(["2020-04-21", "2020-04-22"]))
print(momentos(sin_abr20).apply(lambda v: f"{v:.5g}").to_string())

z = (r - r.mean()) / r.std()
print(f"\nP(|z| > 4): observada = {np.mean(np.abs(z) > 4):.4g} | normal = {2*stats.norm.cdf(-4):.4g}")

# ---- Figura: histograma en escala logarítmica y gráfico QQ ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
malla = np.linspace(-6, 6, 400)
ax[0].hist(z, bins=np.arange(-25, 16.5, 0.25), density=True, color=eg.AZUL, alpha=0.75, label="Brent (estandarizado)")
ax[0].plot(malla, stats.norm.pdf(malla), color=eg.LADRILLO, lw=1.4, label="Normal(0,1)")
ax[0].set_yscale("log"); ax[0].set_ylim(1e-5, 1); ax[0].set_xlim(-12, 12)
ax[0].annotate(f"21-abr-2020: $z$ = {z.min():.1f}\n(fuera del gráfico)".replace(".", ","), xy=(-12, 2e-4),
               xytext=(-11.5, 2e-2), fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.7))
ax[0].set_xlabel("rendimiento estandarizado $z$")
ax[0].set_ylabel("densidad (escala log)"); ax[0].legend(loc="upper right", fontsize=8)
(osm, osr), _ = stats.probplot(z, dist="norm")
ax[1].plot(osm, osr, ".", ms=2, color=eg.AZUL)
ax[1].plot([-5, 5], [-5, 5], color=eg.LADRILLO, lw=1.2, ls="--")
ax[1].set_xlabel("cuantiles de la normal"); ax[1].set_ylabel("cuantiles observados")
for a in ax: eg.coma(a)
ax[0].yaxis.set_major_formatter(plt.LogFormatterMathtext())
fig.savefig("graficos/cap02_fig_normalidad.pdf")
