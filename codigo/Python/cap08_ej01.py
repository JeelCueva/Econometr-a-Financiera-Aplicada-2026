# Capítulo 8, Ejercicio 8.1: ¿hay efectos ARCH? Ljung-Box sobre r y r^2 y prueba LM de Engle
import numpy as np, pandas as pd, statsmodels.api as sm, warnings
warnings.simplefilter("ignore", FutureWarning)
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch   # solo para comprobar

b = pd.read_csv("datos/internacional/brent_diario.csv")
s = pd.read_csv("datos/internacional/sp500_mensual_shiller.csv")
series = {"Brent diario": 100 * np.diff(np.log(b["Price"].to_numpy())),
          "S&P 500 mensual": 100 * np.diff(np.log(s["SP500"].to_numpy()))}

def arch_lm(e, q):                               # regresión de e_t^2 sobre q rezagos de e^2
    e2 = e**2; n = len(e2)
    X = sm.add_constant(np.column_stack([e2[q - j:n - j] for j in range(1, q + 1)]))
    R2 = sm.OLS(e2[q:], X).fit().rsquared
    return (n - q) * R2, stats.chi2.sf((n - q) * R2, q), R2

def ljung_box(x, m):
    x = x - x.mean(); n = len(x); r = acf(x, nlags=m, fft=False)[1:]
    Q = n * (n + 2) * (r**2 / (n - np.arange(1, m + 1))).sum()
    return Q, stats.chi2.sf(Q, m)

for nm, r in series.items():
    e = r - r.mean()
    Q1, p1 = ljung_box(e, 10); Q2, p2 = ljung_box(e**2, 10); LM, pl, R2 = arch_lm(e, 5)
    print(f"{nm} (T = {len(r)})")
    print("  ACF de r   (1-5):", " ".join(f"{v:6.3f}" for v in acf(e, nlags=5)[1:]))
    print("  ACF de r^2 (1-5):", " ".join(f"{v:6.3f}" for v in acf(e**2, nlags=5)[1:]))
    print(f"  Q(10) de r   = {Q1:8.2f}  p = {p1:.2e}")
    print(f"  Q(10) de r^2 = {Q2:8.2f}  p = {p2:.2e}")
    print(f"  LM de Engle (5 rezagos): R2 = {R2:.4f}, LM = {LM:.2f}, p = {pl:.2e}")
    chk = het_arch(e, nlags=5); lb = acorr_ljungbox(e**2, lags=[10])
    print(f"  comprobación statsmodels: LM = {chk[0]:.2f}; Q(10) de r^2 = {lb['lb_stat'].iloc[0]:.2f}\n")

# ---- Figura: ACF de r y de r^2 para las dos series ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(9.4, 5.0), sharey="row")
for j, (nm, r) in enumerate(series.items()):
    e = r - r.mean(); banda = 1.96 / np.sqrt(len(e)); k = np.arange(1, 31)
    for i, (x, et) in enumerate([(e, "$r_t$"), (e**2, "$r_t^2$")]):
        a = ax[i, j]; v = acf(x, nlags=30)[1:]
        a.vlines(k, 0, v, color=eg.AZUL if i == 0 else eg.LADRILLO, lw=1.6)
        a.axhline(banda, color=eg.GRIS, ls=":"); a.axhline(-banda, color=eg.GRIS, ls=":"); a.axhline(0, color="black", lw=0.6)
        a.set_title(f"{nm}: ACF de {et}"); eg.coma(a)
    ax[1, j].set_xlabel("rezago")
fig.tight_layout(); fig.savefig("graficos/cap08_ej01_acf.pdf")
