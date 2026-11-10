# Capítulo 3: gráficas de la teoría (datos simulados o parámetros ilustrativos, semilla fija)
import numpy as np
import estilo_graficos as eg
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)

# --- Figura: MCO minimiza la suma de residuos al cuadrado ---
x = rng.normal(0, 1, 14); y = 0.5 + 0.9 * x + rng.normal(0, 0.6, 14)
b = np.cov(x, y)[0, 1] / x.var(ddof=1); a = y.mean() - b * x.mean()
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
g = np.linspace(x.min() - 0.3, x.max() + 0.3, 2)
for k, (aa, bb, tit, col) in enumerate([(a, b, "recta MCO", eg.LADRILLO), (a + 0.5, 0.3, "otra recta cualquiera", eg.GRIS)]):
    ax[k].plot(g, aa + bb * g, color=col, lw=1.5)
    ax[k].vlines(x, aa + bb * x, y, color=col, lw=0.9, ls=":")
    ax[k].scatter(x, y, color=eg.AZUL, s=16, zorder=3)
    src = np.sum((y - aa - bb * x)**2)
    ax[k].set_title(f"{tit}: suma de residuos² = {src:.2f}".replace(".", ","))
    ax[k].set_xlabel("$x_t$ (exceso del mercado)"); eg.coma(ax[k])
ax[0].set_ylabel("$y_t$ (exceso del activo)")
fig.savefig("graficos/cap03_fig_mco.pdf")

# --- Figura: mismo beta, distinto riesgo idiosincrático ---
T = 300; xm = rng.normal(0, 4.5, T)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for k, (sd_e, nom) in enumerate([(2.0, "activo A"), (9.0, "activo B")]):
    yy = 1.0 * xm + rng.normal(0, sd_e, T)
    r2 = np.corrcoef(xm, yy)[0, 1]**2
    ax[k].scatter(xm, yy, s=5, color=eg.AZUL, alpha=0.5)
    ax[k].plot([-15, 15], [-15, 15], color=eg.LADRILLO, lw=1.4)
    ax[k].set_title(f"{nom}: $\\beta=1$, $\\sigma_\\varepsilon={sd_e:.0f}$ %, $R^2={r2:.2f}$".replace(".", ","))
    ax[k].set_xlabel("exceso del mercado (%)"); eg.coma(ax[k])
ax[0].set_ylabel("exceso del activo (%)"); ax[0].set_ylim(-35, 35)
fig.savefig("graficos/cap03_fig_riesgo.pdf")

# --- Figura: frontera, recta del mercado de capitales y SML (parámetros ilustrativos) ---
mu = np.array([0.04, 0.07, 0.10]); sd = np.array([0.10, 0.16, 0.24])
corr = np.array([[1, 0.3, 0.2], [0.3, 1, 0.4], [0.2, 0.4, 1]]); S = np.outer(sd, sd) * corr
Si = np.linalg.inv(S); u = np.ones(3); A, B, C = u @ Si @ u, u @ Si @ mu, mu @ Si @ mu
m = np.linspace(0.0, 0.14, 300); sdf = np.sqrt((A * m**2 - 2 * B * m + C) / (A * C - B**2))
wq = Si @ mu / (u @ Si @ mu); mq, sq = wq @ mu, np.sqrt(wq @ S @ wq)
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
ax[0].plot(100 * sdf, 100 * m, color=eg.AZUL, label="frontera de activos riesgosos")
xs = np.linspace(0, 30, 2); ax[0].plot(xs, xs * mq / sq, color=eg.LADRILLO, ls="--", label="recta del mercado de capitales")
ax[0].scatter(100 * sd, 100 * mu, color=eg.GRIS, s=18, zorder=3, label="activos individuales")
ax[0].scatter(100 * sq, 100 * mq, marker="*", s=130, color=eg.LADRILLO, zorder=4, label="portafolio tangente $q$")
ax[0].set_xlim(0, 30); ax[0].set_ylim(0, 14); ax[0].set_xlabel("volatilidad $\\sigma$ (%)")
ax[0].set_ylabel("exceso de rendimiento esperado (%)"); ax[0].legend(fontsize=7.5, loc="upper left")
ax[0].set_title("Plano media-volatilidad")
bb = np.linspace(0, 2, 2); prima = 6.0
ax[1].plot(bb, prima * bb, color=eg.LADRILLO, lw=1.5, label="SML: $\\mu_i-R_f=\\beta_i(\\mu_m-R_f)$")
pts = [(0.6, 6.0, "$\\alpha>0$: subvalorado"), (1.4, 5.6, "$\\alpha<0$: sobrevalorado")]
for bx, my, txt in pts:
    ax[1].scatter(bx, my, color=eg.AZUL, s=22, zorder=3)
    ax[1].vlines(bx, prima * bx, my, color=eg.AZUL, ls=":", lw=1)
    ax[1].annotate(txt, (bx, my), xytext=(-95, 8) if my > prima * bx else (8, -4), textcoords="offset points", fontsize=8)
ax[1].scatter(1, prima, marker="D", color=eg.OLIVA, s=26, zorder=3, label="mercado ($\\beta=1$)")
ax[1].set_xlim(0, 2); ax[1].set_ylim(0, 12.5); ax[1].set_xlabel("$\\beta_i$")
ax[1].set_title("Recta del mercado de valores"); ax[1].legend(fontsize=7.5, loc="upper left")
for a in ax: eg.coma(a)
fig.savefig("graficos/cap03_fig_capm.pdf")
