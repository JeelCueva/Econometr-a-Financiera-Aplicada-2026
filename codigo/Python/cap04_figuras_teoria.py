# Capítulo 4: gráficas de la teoría (simulaciones de Monte Carlo con semilla fija)
import numpy as np
import estilo_graficos as eg
import matplotlib.pyplot as plt

rng = np.random.default_rng(2024)

def mco(y, X):
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; return b, y - X @ b, XtXi

def ee_clasico(e, XtXi, k):
    return np.sqrt(np.diag(XtXi) * (e @ e) / (len(e) - k))

def ee_hac(X, e, XtXi, L):              # L = 0 da el estimador HC0 de White
    u = X * e[:, None]; S = u.T @ u
    for j in range(1, L + 1):
        G = u[j:].T @ u[:-j]; S += (1 - j / (L + 1)) * (G + G.T)
    return np.sqrt(np.diag(XtXi @ S @ XtXi))

# --- Figura: heterocedasticidad y tamaño de la prueba t ---
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))
x = rng.standard_normal(400); y = 1.0 * x + (0.3 + 1.2 * np.abs(x)) * rng.standard_normal(400)
ax[0].scatter(x, y, s=6, color=eg.AZUL, alpha=0.6)
ax[0].plot([-3, 3], [-3, 3], color=eg.LADRILLO, lw=1.4)
ax[0].set_title("Varianza del error creciente con $|x_t|$"); ax[0].set_xlabel("$x_t$"); ax[0].set_ylabel("$y_t$")
Ts, reps = [50, 100, 250, 500, 1000], 4000
tasa_c, tasa_r = [], []
for T in Ts:
    rc = rr = 0
    for _ in range(reps):
        x = rng.standard_normal(T); y = (0.3 + 1.2 * np.abs(x)) * rng.standard_normal(T)   # beta verdadera = 0
        X = np.column_stack([np.ones(T), x]); b, e, XtXi = mco(y, X)
        rc += abs(b[1] / ee_clasico(e, XtXi, 2)[1]) > 1.96
        rr += abs(b[1] / (ee_hac(X, e, XtXi, 0)[1] * np.sqrt(T / (T - 2)))) > 1.96   # HC1
    tasa_c.append(rc / reps); tasa_r.append(rr / reps)
ax[1].plot(Ts, 100 * np.array(tasa_c), "o-", color=eg.LADRILLO, label="errores clásicos")
ax[1].plot(Ts, 100 * np.array(tasa_r), "s--", color=eg.AZUL, label="errores robustos (HC1)")
ax[1].axhline(5, color=eg.GRIS, ls=":", label="nivel nominal 5 %")
ax[1].set_xscale("log"); ax[1].set_xticks(Ts); ax[1].set_xticklabels(Ts); ax[1].set_ylim(0, 25)
ax[1].xaxis.set_minor_formatter(plt.NullFormatter())
ax[1].set_title("Rechazos de $H_0:\\beta=0$ siendo cierta"); ax[1].set_xlabel("$T$ (escala log)")
ax[1].set_ylabel("% de rechazos"); ax[1].legend(fontsize=8)
eg.coma(ax[0]); ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
fig.savefig("graficos/cap04_fig_hetero.pdf")
print("Heterocedasticidad, % de rechazos (clásico):", np.round(100 * np.array(tasa_c), 1))
print("Heterocedasticidad, % de rechazos (HC1):    ", np.round(100 * np.array(tasa_r), 1))

# --- Figura: rendimientos traslapados ---
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))
k = 12; j = np.arange(0, 25)
ax[0].vlines(j, 0, np.clip((k - j) / k, 0, None), color=eg.AZUL, lw=2)
ax[0].set_title(f"Autocorrelación de sumas traslapadas de $k={k}$ meses")
ax[0].set_xlabel("rezago $j$ (meses)"); ax[0].set_ylabel("$\\rho_j=(k-j)/k$"); eg.coma(ax[0])
T, reps, phi = 600, 2000, 0.98
rech = {"clásico": 0, "Newey-West, $L=12$": 0, "sin traslape (anual)": 0}
for _ in range(reps):
    g = rng.standard_normal(T + k)                          # rendimientos mensuales i.i.d.
    z = np.zeros(T); v = rng.standard_normal(T)
    for t in range(1, T): z[t] = phi * z[t - 1] + v[t]      # predictor persistente e independiente
    y = np.array([g[t + 1:t + 1 + k].sum() for t in range(T)])
    X = np.column_stack([np.ones(T), z]); b, e, XtXi = mco(y, X)
    rech["clásico"] += abs(b[1] / ee_clasico(e, XtXi, 2)[1]) > 1.96
    rech["Newey-West, $L=12$"] += abs(b[1] / ee_hac(X, e, XtXi, 12)[1]) > 1.96
    idx = np.arange(0, T, k); bb, ee_, XX = mco(y[idx], X[idx])
    rech["sin traslape (anual)"] += abs(bb[1] / ee_clasico(ee_, XX, 2)[1]) > 1.96
tasas = {n: 100 * v / reps for n, v in rech.items()}
ax[1].bar(range(3), list(tasas.values()), color=[eg.LADRILLO, eg.AZUL, eg.OLIVA], width=0.6)
ax[1].axhline(5, color=eg.GRIS, ls=":")
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(list(tasas.keys()), fontsize=8)
for i, v in enumerate(tasas.values()): ax[1].text(i, v + 1, f"{v:.1f} %".replace(".", ","), ha="center", fontsize=8)
ax[1].set_ylabel("% de rechazos de $H_0$ cierta"); ax[1].set_title("Prueba $t$ al 5 % con $T=600$ meses")
ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
fig.savefig("graficos/cap04_fig_traslape.pdf")
print("Traslape, % de rechazos:", {n: round(v, 1) for n, v in tasas.items()})

# --- Figura: un cambio estructural ---
fig, ax = plt.subplots(figsize=(5.8, 3.6))
x1, x2 = rng.normal(0, 1, 150), rng.normal(0, 1, 150)
y1, y2 = 0.9 * x1 + rng.normal(0, 0.5, 150), 0.3 * x2 + rng.normal(0, 0.5, 150)
ax.scatter(x1, y1, s=8, color=eg.AZUL, alpha=0.6, label="régimen 1 ($\\beta=0{,}9$)")
ax.scatter(x2, y2, s=8, color=eg.LADRILLO, marker="^", alpha=0.6, label="régimen 2 ($\\beta=0{,}3$)")
xx = np.linspace(-3, 3, 2); bp = np.polyfit(np.r_[x1, x2], np.r_[y1, y2], 1)
ax.plot(xx, 0.9 * xx, color=eg.AZUL); ax.plot(xx, 0.3 * xx, color=eg.LADRILLO)
ax.plot(xx, np.polyval(bp, xx), color="black", ls="--", label=f"una sola recta: $\\hat\\beta={bp[0]:.2f}$".replace(".", ","))
ax.set_xlabel("$x_t$"); ax.set_ylabel("$y_t$"); ax.legend(fontsize=8); eg.coma(ax)
fig.savefig("graficos/cap04_fig_quiebre.pdf")
print("Recta común (simulación):", round(bp[0], 3))
