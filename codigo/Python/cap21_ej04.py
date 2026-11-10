# Capítulo 21, Etapa 4: VaR y ES de la cartera 1/N y su prueba retrospectiva (2016-2018)
import numpy as np
from cap21_funciones import cargar, garch_fit, garch_s2, ewma_var, var_es_t, var_es_normal, var_es_hist, kupiec, christoffersen

f, R = cargar(); rp = R[:, :4].mean(1); n = len(rp); ne = int((f < "2016-01-01").sum()); L = -rp
g = garch_fit(rp[:ne], "t"); p = g["p"]; s2 = garch_s2(p, rp, g["bc"]); he = ewma_var(rp)
PA = {}                                                   # reestimación cada enero con todos los datos anteriores
for anio in ("2016", "2017", "2018"):
    i0 = int((f < anio + "-01-01").sum()); i1 = int((f < str(int(anio) + 1) + "-01-01").sum()); ga = garch_fit(rp[:i0], "t")
    sa = garch_s2(ga["p"], rp[:i1], ga["bc"]); PA[anio] = ga["p"]
    for t in range(i0, i1): PA[t] = (ga["p"], sa[t])
print(f"Cartera 1/N (rebalanceo diario); GARCH(1,1)-t estimado en 2014-2015: mu = {p[0]:.4f}, omega = {p[1]:.4f}, alfa = {p[2]:.4f}, beta = {p[3]:.4f}, nu = {p[4]:.2f}")
for a in ("2016", "2017", "2018"): print(f"  reestimado con datos hasta {int(a) - 1}: alfa = {PA[a][2]:.4f}, beta = {PA[a][3]:.4f}, nu = {PA[a][4]:.2f}")
print(f"Prueba retrospectiva: {f[ne]} a {f[-1]} ({n - ne} días)\n")
print("nivel   excepciones  esperadas   Kupiec LR   p       Christoffersen LR   p      pérdida/ES en excepciones   método")
for nivel in (0.99, 0.975):
    M = {}
    M["simulación histórica (250 días)"] = np.array([var_es_hist(L[t - 250:t], nivel) for t in range(ne, n)])
    M["EWMA normal"] = np.array([var_es_normal(0.0, np.sqrt(he[t]), nivel) for t in range(ne, n)])
    M["GARCH(1,1)-t"] = np.array([var_es_t(p[0], np.sqrt(s2[t]), p[4], nivel) for t in range(ne, n)])
    M["GARCH(1,1)-t reestimado cada año"] = np.array([var_es_t(PA[t][0][0], np.sqrt(PA[t][1]), PA[t][0][4], nivel) for t in range(ne, n)])
    for nom, VE in M.items():
        x = (L[ne:] > VE[:, 0]).astype(int); k = x.sum(); lr, pk = kupiec(x, nivel); lc, pc, _ = christoffersen(x)
        ratio = L[ne:][x == 1].mean() / VE[x == 1, 1].mean() if k else np.nan
        print(f"{100 * nivel:5.1f}   {k:8d}   {(1 - nivel) * (n - ne):10.1f}   {lr:8.2f}  {pk:6.4f}   {lc:12.2f}   {pc:6.4f}   {ratio:18.3f}        {nom}")
    if nivel == 0.99: G99 = M
gf = garch_fit(rp, "t"); q = gf["p"]; sf = gf["s2"]; VaRf = var_es_t(q[0], np.sqrt(q[1] + q[2] * (rp[-1] - q[0])**2 + q[3] * sf[-1]), q[4], 0.99)
print(f"\nGARCH(1,1)-t con toda la muestra: alfa = {q[2]:.4f}, beta = {q[3]:.4f}, nu = {q[4]:.2f}")
print(f"Pronóstico para el día siguiente a {f[-1]}: volatilidad = {np.sqrt(q[1] + q[2] * (rp[-1] - q[0])**2 + q[3] * sf[-1]):.4f} %; VaR 99 % = {VaRf[0]:.4f} %; ES 99 % = {VaRf[1]:.4f} %")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8.6, 3.4)); t = np.arange(ne, n)
ax.plot(t, L[ne:], color=eg.GRIS, lw=0.5, label="pérdida diaria de la cartera")
for nom, c, ls in [("GARCH(1,1)-t", eg.LADRILLO, "-"), ("simulación histórica (250 días)", eg.AZUL, "--")]:
    ax.plot(t, G99[nom][:, 0], color=c, ls=ls, lw=1.0, label=f"VaR 99 %, {nom}")
    x = L[ne:] > G99[nom][:, 0]; ax.plot(t[x], L[ne:][x], "o", ms=3.5, mfc="none", color=c)
pos = [k for k in t if f[k][:4] != f[k - 1][:4]]; ax.set_xticks(pos); ax.set_ylabel("%"); ax.legend(fontsize=7, loc="upper left")
eg.coma(ax); ax.set_xticklabels([f[k][:4] for k in pos]); fig.savefig("graficos/cap21_ej04_var.pdf")
