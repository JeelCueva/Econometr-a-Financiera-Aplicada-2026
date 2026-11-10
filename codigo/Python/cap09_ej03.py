# Capítulo 9, Ejercicio 9.3: EGARCH-t para el S&P 500, comparación de modelos y curvas de impacto de noticias
import numpy as np, pandas as pd
from cap09_funciones import ajustar, RAIZ_2_PI
from arch import arch_model                              # solo para comprobar

s = pd.read_csv("datos/internacional/sp500_diario_yahoo.csv"); r = 100 * np.diff(np.log(s["Adj Close"].to_numpy())); T = len(r)
m = {"GARCH": ajustar(r, "garch", "c", "t"), "GJR": ajustar(r, "gjr", "c", "t"), "EGARCH": ajustar(r, "egarch", "c", "t")}
par = {k: dict(zip(v["nombres"], v["p"])) for k, v in m.items()}; p = par["EGARCH"]
print("EGARCH(1,1)-t:")
tab = pd.DataFrame({"coef": m["EGARCH"]["p"], "ee_rob": m["EGARCH"]["ee_rob"]}, index=m["EGARCH"]["nombres"])
tab["t_rob"] = (tab.coef / tab.ee_rob).round(2); print(tab.round(4))
print(f"pendiente de g(z): z > 0: alpha + gamma = {p['alpha'] + p['gamma']:.4f}; z < 0: gamma - alpha = {p['gamma'] - p['alpha']:.4f}")
print(f"vida media (en logaritmos) = {np.log(0.5) / np.log(p['beta']):.1f} días\n")
print("Modelo     k    log-ver        AIC        BIC")
for nm, v in m.items():
    print(f"{nm:<8} {v['k']:3d} {v['ll']:10.2f} {-2 * v['ll'] + 2 * v['k']:10.2f} {-2 * v['ll'] + v['k'] * np.log(T):10.2f}")

def nic(mod, x, s2):                                     # varianza de mañana como función del choque de hoy
    q = par[mod]
    if mod == "EGARCH":
        z = x / np.sqrt(s2)
        return np.exp(q["omega"] + q["alpha"] * (np.abs(z) - RAIZ_2_PI) + q["gamma"] * z + q["beta"] * np.log(s2))
    g = q.get("gamma", 0.0)
    return q["omega"] + (q["alpha"] + g * (x < 0)) * x**2 + q["beta"] * s2
s2 = r.var(ddof=1); x = 2 * np.sqrt(s2)
print(f"\nCurva de impacto con sigma2_{{t-1}} = varianza muestral ({s2:.4f}); choque de +-2 desviaciones ({x:.3f} %):")
for nm in m:
    a, b = nic(nm, -x, s2), nic(nm, x, s2)
    print(f"  {nm:<7} sigma2(-) = {a:.4f}  sigma2(+) = {b:.4f}  cociente = {a / b:.2f}")
a = arch_model(r, vol="EGARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
print(f"\ncomprobación con arch (EGARCH-t): {np.round(a.params.to_numpy(), 4)}, log-ver = {a.loglikelihood:.2f}")

# ---- Figura: curvas de impacto de noticias estimadas ----
import estilo_graficos as eg, matplotlib.pyplot as plt
xs = np.linspace(-6, 6, 400)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for nm, col, ls in [("GARCH", eg.GRIS, ":"), ("GJR", eg.LADRILLO, "-"), ("EGARCH", eg.AZUL, "--")]:
    ax.plot(xs, nic(nm, xs, s2), color=col, ls=ls, lw=1.4, label=f"{nm}-$t$")
ax.axvline(0, color="black", lw=0.5)
ax.set_xlabel("choque de hoy $\\varepsilon_{t-1}$ (%)"); ax.set_ylabel("varianza de mañana $\\sigma_t^2$")
ax.set_title("S&P 500: curvas de impacto de noticias ($\\sigma^2_{t-1}$ = varianza muestral)")
ax.legend(fontsize=8); eg.coma(ax); fig.savefig("graficos/cap09_ej03_nic_sp500.pdf")
