# Capítulo 5: gráficas de la teoría (simulaciones con semilla fija)
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf
import estilo_graficos as eg
import matplotlib.pyplot as plt

rng = np.random.default_rng(55)

def ar1(phi, T, rng):
    e = rng.standard_normal(T); y = np.zeros(T)
    for t in range(1, T): y[t] = phi * y[t - 1] + e[t]
    return y

def t_df(y, det="c"):                       # estadístico t de Dickey-Fuller sin rezagos
    dy, y1 = np.diff(y), y[:-1]; n = len(dy)
    cols = [y1] + ([np.ones(n)] if det in ("c", "ct") else []) + ([np.arange(n)] if det == "ct" else [])
    X = np.column_stack(cols); b, *_ = np.linalg.lstsq(X, dy, rcond=None); e = dy - X @ b
    s2 = e @ e / (n - X.shape[1]); return b[0] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])

# --- Figura: cuatro procesos y sus autocorrelaciones ---
T = 500
casos = [("ruido blanco", 0.0), ("AR(1), $\\phi=0{,}5$", 0.5), ("AR(1), $\\phi=0{,}95$", 0.95), ("paseo aleatorio, $\\phi=1$", 1.0)]
fig, ax = plt.subplots(4, 2, figsize=(9.4, 7.2), gridspec_kw={"width_ratios": [2.4, 1]})
for i, (nom, phi) in enumerate(casos):
    y = ar1(phi, T, rng)
    ax[i, 0].plot(y, color=eg.AZUL if phi < 1 else eg.LADRILLO, lw=0.7); ax[i, 0].set_ylabel(nom, fontsize=8)
    r = acf(y, nlags=40)[1:]
    ax[i, 1].vlines(range(1, 41), 0, r, color=eg.AZUL if phi < 1 else eg.LADRILLO)
    if phi > 0: ax[i, 1].plot(range(1, 41), phi ** np.arange(1, 41), color=eg.GRIS, ls="--", lw=1)
    ax[i, 1].axhline(0, color="black", lw=0.5); ax[i, 1].set_ylim(-0.2, 1.05)
    for a in ax[i]: eg.coma(a)
ax[0, 0].set_title("Trayectoria simulada ($T=500$)"); ax[0, 1].set_title("ACF muestral y teórica $\\phi^j$")
ax[3, 0].set_xlabel("$t$"); ax[3, 1].set_xlabel("rezago $j$")
fig.tight_layout(); fig.savefig("graficos/cap05_fig_procesos.pdf")

# --- Figura: distribución de Dickey-Fuller ---
reps, T = 20000, 500
res = {d: np.empty(reps) for d in ("n", "c", "ct")}
for r in range(reps):
    y = np.cumsum(rng.standard_normal(T))
    for d in res: res[d][r] = t_df(y, d)
print("Cuantiles simulados del estadístico t de Dickey-Fuller (T = 500, 20 000 réplicas)")
for d, nom in [("n", "sin constante"), ("c", "con constante"), ("ct", "con constante y tendencia")]:
    print(f"  {nom:<26} 1 %: {np.quantile(res[d], .01):6.3f}  5 %: {np.quantile(res[d], .05):6.3f}  10 %: {np.quantile(res[d], .10):6.3f}")
fig, ax = plt.subplots(figsize=(7, 3.6))
g = np.linspace(-6, 4, 400)
ax.plot(g, stats.norm.pdf(g), color="black", ls=":", label="normal estándar")
for d, col, ls, nom in [("n", eg.GRIS, "-.", "DF sin constante"), ("c", eg.AZUL, "-", "DF con constante"), ("ct", eg.LADRILLO, "--", "DF con constante y tendencia")]:
    ax.plot(g, stats.gaussian_kde(res[d])(g), color=col, ls=ls, label=nom)
ax.axvline(np.quantile(res["c"], .05), color=eg.AZUL, lw=0.8)
ax.annotate("5 % con constante", xy=(np.quantile(res["c"], .05), 0.02), xytext=(-5.8, 0.3), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.7))
ax.axvline(-1.645, color="black", lw=0.8, ls=":")
ax.set_xlabel("estadístico $t$ de $\\hat\\gamma$ bajo $H_0$: raíz unitaria"); ax.set_ylabel("densidad")
ax.legend(fontsize=8, loc="upper right"); eg.coma(ax)
fig.savefig("graficos/cap05_fig_df.pdf")

# --- Figura: regresión espuria ---
Ts, reps = [50, 100, 250, 500, 1000], 2000
niv, dif = [], []
for T in Ts:
    a = b = 0
    for _ in range(reps):
        x, y = np.cumsum(rng.standard_normal(T)), np.cumsum(rng.standard_normal(T))
        for serie_x, serie_y, cont in [(x, y, "n"), (np.diff(x), np.diff(y), "d")]:
            X = np.column_stack([np.ones(len(serie_x)), serie_x]); bb, *_ = np.linalg.lstsq(X, serie_y, rcond=None)
            e = serie_y - X @ bb; s2 = e @ e / (len(e) - 2); t = bb[1] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
            if abs(t) > 1.96:
                if cont == "n": a += 1
                else: b += 1
    niv.append(100 * a / reps); dif.append(100 * b / reps)
print("Regresión espuria, % de |t| > 1,96 (niveles):  ", np.round(niv, 1))
print("Regresión espuria, % de |t| > 1,96 (diferencias):", np.round(dif, 1))
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
while True:                                  # ejemplo ilustrativo: un par con R2 alto
    x, y = np.cumsum(rng.standard_normal(300)), np.cumsum(rng.standard_normal(300))
    if np.corrcoef(x, y)[0, 1]**2 > 0.5: break
ax[0].plot(x, color=eg.AZUL, lw=0.9, label="$x_t$"); ax[0].plot(y, color=eg.LADRILLO, lw=0.9, ls="--", label="$y_t$")
r2 = np.corrcoef(x, y)[0, 1]**2
ax[0].set_title(f"Dos paseos aleatorios independientes ($R^2={r2:.2f}$)".replace(".", ",")); ax[0].legend(fontsize=8); ax[0].set_xlabel("$t$")
ax[1].plot(Ts, niv, "o-", color=eg.LADRILLO, label="regresión en niveles")
ax[1].plot(Ts, dif, "s--", color=eg.AZUL, label="regresión en diferencias")
ax[1].axhline(5, color=eg.GRIS, ls=":"); ax[1].set_xscale("log"); ax[1].set_xticks(Ts); ax[1].set_xticklabels(Ts)
ax[1].xaxis.set_minor_formatter(plt.NullFormatter()); ax[1].set_ylim(0, 100)
ax[1].set_xlabel("$T$ (escala log)"); ax[1].set_ylabel("% de rechazos de $\\beta=0$"); ax[1].legend(fontsize=8)
ax[1].set_title("La prueba $t$ al 5 % entre paseos independientes")
for a in ax: eg.coma(a)
ax[1].xaxis.set_major_formatter(plt.FixedFormatter([str(t) for t in Ts]))
fig.savefig("graficos/cap05_fig_espuria.pdf")
print(f"R2 del ejemplo de la figura: {r2:.3f}")

# --- Figura: el efecto Perron ---
T, reps = 300, 2000
rech_sin, rech_con = 0, 0
for _ in range(reps):
    y = ar1(0.9, T, rng) + np.where(np.arange(T) >= T // 2, 10.0, 0.0)
    rech_sin += t_df(y, "c") < -2.87
    yc = y - np.where(np.arange(T) >= T // 2, y[T // 2:].mean() - y[:T // 2].mean(), 0.0)   # quita el salto
    rech_con += t_df(yc, "c") < -2.87
print(f"Efecto Perron: ADF rechaza la raíz unitaria {100*rech_sin/reps:.1f} % de las veces con el salto "
      f"y {100*rech_con/reps:.1f} % sin él")
fig, ax = plt.subplots(figsize=(7, 3))
y = ar1(0.9, T, rng) + np.where(np.arange(T) >= T // 2, 10.0, 0.0)
ax.plot(y, color=eg.AZUL, lw=0.8)
ax.hlines([0, 10], [0, T // 2], [T // 2, T], color=eg.LADRILLO, lw=1.5, label="media de cada régimen")
ax.set_xlabel("$t$"); ax.legend(fontsize=8, loc="upper left"); eg.coma(ax)
ax.set_title("AR(1) estacionario con $\\phi=0{,}9$ y un cambio de nivel en $t=150$")
fig.savefig("graficos/cap05_fig_perron.pdf")

# --- Figura: idea de la prueba KPSS (sumas parciales) ---
T = 500
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.2))
for i, (phi, nom) in enumerate([(0.5, "estacionaria (AR(1), $\\phi=0{,}5$)"), (1.0, "paseo aleatorio")]):
    y = ar1(phi, T, rng); e = y - y.mean(); S = np.cumsum(e)
    ax[i].plot(S, color=eg.AZUL if phi < 1 else eg.LADRILLO, lw=0.9)
    ax[i].axhline(0, color="black", lw=0.5); ax[i].set_title(f"Sumas parciales $S_t$: serie {nom}"); ax[i].set_xlabel("$t$"); eg.coma(ax[i])
fig.savefig("graficos/cap05_fig_kpss.pdf")
