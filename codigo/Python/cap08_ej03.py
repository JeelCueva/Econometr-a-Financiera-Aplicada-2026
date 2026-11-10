# Capítulo 8, Ejercicio 8.3: GARCH(1,1) con innovaciones t de Student y diagnóstico de residuos estandarizados
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from cap08_funciones import garch_fit
from arch import arch_model                        # solo para comprobar

b = pd.read_csv("datos/internacional/brent_diario.csv"); r = 100 * np.diff(np.log(b["Price"].to_numpy())); T = len(r)
mn = garch_fit(r, "normal"); mt = garch_fit(r, "t"); p = mt["par"]
print("GARCH(1,1)-t:")
tab = pd.DataFrame({"coef": p, "ee_rob": mt["ee_rob"]}); tab["t_rob"] = (tab["coef"] / tab["ee_rob"]).round(2)
print(tab.round(4))
crit = lambda ll, k: (ll, -2 * ll + 2 * k, -2 * ll + k * np.log(T))
print("\n             log-ver        AIC        BIC")
for nm, ll, k in [("normal", mn["ll"], 4), ("t", mt["ll"], 5)]:
    print(f"{nm:<7} {crit(ll, k)[0]:12.2f} {crit(ll, k)[1]:10.2f} {crit(ll, k)[2]:10.2f}")
pers = p["alpha"] + p["beta"]
print(f"\npersistencia = {pers:.5f}; vida media = {np.log(0.5) / np.log(pers):.1f} días; "
      f"curtosis de la t estandarizada = {3 + 6 / (p['nu'] - 4):.2f}")

def diagn(x, nm):
    e = x - x.mean(); s = (e**3).mean() / (e**2).mean()**1.5; k = (e**4).mean() / (e**2).mean()**2
    jb = len(x) / 6 * (s**2 + (k - 3)**2 / 4)
    q1 = acorr_ljungbox(x, lags=[10]).iloc[0]; q2 = acorr_ljungbox(e**2, lags=[10]).iloc[0]
    print(f"{nm:<22} asim. {s:6.2f}  curt. {k:7.2f}  JB {jb:10.0f}  Q(10) z {q1['lb_stat']:6.2f} (p {q1['lb_pvalue']:.3f})"
          f"  Q(10) z^2 {q2['lb_stat']:8.2f} (p {q2['lb_pvalue']:.3f})")
print("\nDiagnóstico:")
diagn(r, "rendimientos r_t")
z = (r - p["mu"]) / np.sqrt(mt["s2"]); nu = p["nu"]
diagn(z, "residuos z_t (modelo t)")
print(f"\nproporción |z| > 4: observada {np.mean(np.abs(z) > 4):.5f}; normal {2 * stats.norm.sf(4):.5f}; "
      f"t(nu) estandarizada {2 * stats.t.sf(4 * np.sqrt(nu / (nu - 2)), nu):.5f}")
a = arch_model(r, dist="t").fit(disp="off")
print(f"\ncomprobación con arch: nu = {a.params['nu']:.4f}, log-verosimilitud = {a.loglikelihood:.4f}, "
      f"AIC = {a.aic:.2f}, BIC = {a.bic:.2f}")

# ---- Figura: gráficos cuantil-cuantil de z_t frente a la normal y a la t(nu) ----
import estilo_graficos as eg, matplotlib.pyplot as plt
zs = np.sort(z); u = (np.arange(1, T + 1) - 0.5) / T
qn = stats.norm.ppf(u); qt = stats.t.ppf(u, nu) * np.sqrt((nu - 2) / nu)
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8))
for a_, q, tit in [(ax[0], qn, "frente a la normal"), (ax[1], qt, f"frente a la $t$ estandarizada ($\\nu={nu:.2f}$)".replace(".", ","))]:
    a_.plot(q, zs, ".", color=eg.AZUL, ms=2); lim = [-10, 10]
    a_.plot(lim, lim, color=eg.LADRILLO, lw=0.8); a_.set_xlim(lim); a_.set_ylim(-10, 10)
    a_.set_title(f"$z_t$ {tit}"); a_.set_xlabel("cuantil teórico"); eg.coma(a_)
ax[0].set_ylabel("cuantil de $z_t$")
fig.savefig("graficos/cap08_ej03_qq.pdf")
