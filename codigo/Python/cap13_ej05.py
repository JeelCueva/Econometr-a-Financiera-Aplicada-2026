# Capítulo 13, Ejercicio 13.5: VaR a 10 días del Brent: regla de la raíz, varianza GARCH y simulación filtrada
import numpy as np, pandas as pd
from scipy import stats
from cap13_funciones import var_es_t, var_es_hist
from cap08_funciones import garch_fit

b = pd.read_csv("datos/internacional/brent_diario.csv"); r = 100 * np.diff(np.log(b["Price"].to_numpy())); f = b["Date"].to_numpy()[1:]; T = len(r)
m = garch_fit(r, "t"); q = m["par"]; z = (r - q["mu"]) / np.sqrt(m["s2"]); H, p, N = 10, 0.99, 100000
print(f"GARCH(1,1)-t: mu = {q['mu']:.4f}, omega = {q['omega']:.4f}, alpha = {q['alpha']:.4f}, beta = {q['beta']:.4f}, nu = {q['nu']:.3f}")
rng = np.random.default_rng(2026); res = {}
for fecha in ["2019-12-31", "2026-11-03"]:
    t = int(np.flatnonzero(f == fecha)[0])
    s2 = q["omega"] + q["alpha"] * (r[t] - q["mu"])**2 + q["beta"] * m["s2"][t]          # varianza del día siguiente
    v1 = var_es_t(q["mu"], np.sqrt(s2), q["nu"], p)
    phi = q["alpha"] + q["beta"]; sb = q["omega"] / (1 - phi)
    VH = (sb + phi**np.arange(H) * (s2 - sb)).sum()                                     # varianza acumulada a 10 días
    h = np.full(N, s2); acc = np.zeros(N)                                               # simulación histórica filtrada (vectorizada)
    for k in range(H):
        e = np.sqrt(h) * z[rng.integers(0, T, N)]; acc += q["mu"] + e; h = q["omega"] + q["alpha"] * e**2 + q["beta"] * h
    sim = -acc; fh = var_es_hist(sim, p); res[fecha] = sim
    print(f"\n{fecha}: volatilidad diaria de mañana = {np.sqrt(s2):.3f} % (anualizada {np.sqrt(252 * s2):.1f} %)")
    print(f"  VaR 99 % a 1 día (t)                  = {v1[0]:6.2f} %;  ES = {v1[1]:6.2f} %")
    print(f"  regla de la raíz: raíz(10) x VaR      = {np.sqrt(H) * v1[0]:6.2f} %")
    print(f"  varianza GARCH acumulada (normal)     = {-H * q['mu'] + stats.norm.ppf(p) * np.sqrt(VH):6.2f} %   (raíz de la varianza a 10 días = {np.sqrt(VH):.3f} %)")
    print(f"  simulación filtrada ({N} trayectorias) = {fh[0]:6.2f} %;  ES = {fh[1]:6.2f} %")

# ---- Figura: distribución simulada de la pérdida a 10 días ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
for a, (fecha, sim), col in zip(ax, res.items(), [eg.AZUL, eg.LADRILLO]):
    v, e = var_es_hist(sim, p); a.hist(sim, bins=120, density=True, color=col, alpha=0.6)
    a.axvline(v, color="black", lw=1); a.axvline(e, color="black", lw=1, ls="--")
    a.set_title(f"{fecha}: VaR$_{{99}}$ = {v:.1f} %, ES$_{{99}}$ = {e:.1f} %".replace(".", ","), fontsize=9)
    a.set_xlabel("pérdida acumulada a 10 días (%)"); eg.coma(a)
ax[0].set_ylabel("densidad"); fig.tight_layout(); fig.savefig("graficos/cap13_ej05_10dias.pdf")
