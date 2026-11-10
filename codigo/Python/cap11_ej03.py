# Capítulo 11, Ejercicio 11.3: el diferencial WTI-Brent y el cambio de régimen de 2011
import numpy as np, pandas as pd, warnings
from scipy import stats
from cap11_funciones import crudo_mensual, johansen, vecm, lr_beta
from cap05_funciones import adf
warnings.simplefilter("ignore")

m = crudo_mensual(); mes = m.index.to_numpy(); L = np.log(m[["Price_b", "Price_w"]].to_numpy()); s = 100 * (L[:, 1] - L[:, 0])
per = {"1987-05 a 2010-12": mes <= "2010-12", "2011-01 a 2026-10": mes >= "2011-01", "muestra completa": np.ones(len(mes), bool)}
print("Diferencial s = 100 (log WTI - log Brent), en % del precio del Brent")
print("Período               T    media  desv.est.  ADF s (p)        traza r=0  beta_WTI   LR beta=(1,-1)  valor p")
for nm, k in per.items():
    a = adf(s[k], "c"); J = johansen(L[k], 2); v = vecm(L[k], 2); LR, *_ = lr_beta(L[k], 2, np.array([[1.0], [-1.0]]))
    print(f"{nm:<18} {k.sum():4d} {s[k].mean():8.2f} {s[k].std(ddof=1):9.2f} {a['t']:7.2f} ({a['p']}) {J['traza'][0]:14.2f} "
          f"{v['beta'][1, 0]:10.4f} {LR:12.2f} {stats.chi2.sf(LR, 1):12.4f}")
print(f"\nmínimo del diferencial: {s.min():.2f} % ({mes[s.argmin()]}); máximo: {s.max():.2f} % ({mes[s.argmax()]})")
k = (mes >= "2011-01") & (mes <= "2013-12")
print(f"promedio 2011-2013: {s[k].mean():.2f} %; promedio 2014-2026: {s[mes >= '2014-01'].mean():.2f} %")

# ---- Figura: diferencial con las medias de cada régimen ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fe = pd.to_datetime(mes)
fig, ax = plt.subplots(figsize=(9.0, 3.4))
ax.plot(fe, s, color=eg.GRIS, lw=0.9, label="diferencial WTI $-$ Brent")
for nm, col in [("1987-05 a 2010-12", eg.AZUL), ("2011-01 a 2026-10", eg.LADRILLO)]:
    k = per[nm]; ax.plot(fe[k], np.full(k.sum(), s[k].mean()), color=col, lw=1.6, label=f"media {nm}")
ax.axvspan(pd.Timestamp("2011-01-01"), pd.Timestamp("2013-12-31"), color=eg.LADRILLO, alpha=0.08)
ax.axhline(0, color="black", lw=0.5); ax.set_ylabel("% del precio del Brent")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.savefig("graficos/cap11_ej03_diferencial.pdf")
