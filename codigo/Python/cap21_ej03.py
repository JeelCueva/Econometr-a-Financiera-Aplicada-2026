# Capítulo 21, Etapa 3: volatilidad: GARCH(1,1)-t estimado en 2014-2015 y evaluado fuera de muestra en 2016-2018
import numpy as np
from cap21_funciones import cargar, NOMBRES, garch_fit, garch_s2, ewma_var

f, R = cargar(); ne = int((f < "2016-01-01").sum()); n = len(f)
print(f"Estimación: {f[0]} a {f[ne - 1]} ({ne} días); evaluación: {f[ne]} a {f[-1]} ({n - ne} días); parámetros fijos, varianza filtrada día a día")
print("\nserie    mu       omega     alfa     beta     nu     alfa+beta   vida media (días)")
H = {}
for j, nom in enumerate(NOMBRES):
    g = garch_fit(R[:ne, j], "t"); p = g["p"]; s2 = garch_s2(p, R[:, j], g["bc"]); H[nom] = (p, s2)
    print(f"{nom:6s}  {p[0]:6.3f}  {p[1]:8.4f}  {p[2]:7.4f}  {p[3]:7.4f}  {p[4]:6.2f}   {p[2] + p[3]:8.4f}   {np.log(0.5) / np.log(p[2] + p[3]):10.1f}")
print("\nPérdidas fuera de muestra (proxy: rendimiento al cuadrado), promedio diario")
print("serie    QLIKE GARCH-t   QLIKE EWMA   QLIKE constante    ECM GARCH-t   ECM EWMA   ECM constante   DM QLIKE (GARCH - EWMA)")
for j, nom in enumerate(NOMBRES):
    p, s2 = H[nom]; r = R[:, j]; o = slice(ne, n); e2 = (r - p[0])**2
    he = ewma_var(r); hc = np.full(n, R[:ne, j].var(ddof=1))
    q = lambda h: np.log(h[o]) + e2[o] / h[o]; m_ = lambda h: (e2[o] - h[o])**2
    d = q(s2) - q(he); L = 5; dc = d - d.mean(); S = dc @ dc
    for k in range(1, L + 1): S += 2 * (1 - k / (L + 1)) * (dc[k:] @ dc[:-k])
    dm = d.mean() / np.sqrt(S / len(d)**2)
    print(f"{nom:6s}  {q(s2).mean():12.4f}  {q(he).mean():11.4f}  {q(hc).mean():14.4f}   {m_(s2).mean():12.2f}  {m_(he).mean():9.2f}  {m_(hc).mean():13.2f}   {dm:10.2f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8.6, 3.2)); t = np.arange(n); p, s2 = H["SP500"]
ax.plot(t, np.abs(R[:, 4]), color=eg.GRIS, lw=0.5, label="|rendimiento| del S&P 500")
ax.plot(t, np.sqrt(s2), color=eg.LADRILLO, lw=1.0, label="volatilidad GARCH(1,1)-t"); ax.plot(t, np.sqrt(ewma_var(R[:, 4])), color=eg.AZUL, lw=0.9, ls="--", label="volatilidad EWMA")
ax.axvline(ne - 0.5, color="black", lw=0.8, ls=":"); ax.text(ne + 5, ax.get_ylim()[1] * 0.9, "fuera de muestra", fontsize=8)
pos = [k for k in t if (k == 0 or f[k][:4] != f[k - 1][:4])]; ax.set_xticks(pos); ax.set_ylabel("%"); ax.legend(fontsize=8, loc="upper left")
eg.coma(ax); ax.set_xticklabels([f[k][:4] for k in pos]); fig.savefig("graficos/cap21_ej03_volatilidad.pdf")
