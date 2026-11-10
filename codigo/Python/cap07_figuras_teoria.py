# Capítulo 7: gráficas de la teoría (curvas de razón de varianzas y tamaño de la prueba con volatilidad GARCH)
import numpy as np
import estilo_graficos as eg
import matplotlib.pyplot as plt

# --- Figura: razón de varianzas teórica ---
q = np.arange(1, 61)
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))
for rho, col, ls in [(0.15, eg.LADRILLO, "-"), (0.0, "black", ":"), (-0.15, eg.AZUL, "--")]:
    vr = (q * (1 - rho**2) - 2 * rho * (1 - rho**q)) / (1 - rho)**2 / q     # AR(1) en rendimientos
    ax[0].plot(q, vr, color=col, ls=ls, label=f"$\\rho = {rho:.2f}$".replace(".", ","))
ax[0].set_title("Rendimientos AR(1)"); ax[0].set_xlabel("horizonte $q$"); ax[0].set_ylabel("$VR(q)$"); ax[0].legend(fontsize=8)
s2w = 1.0
q = np.arange(1, 241)
for phi, s2z, col, ls in [(0.95, 2.0, eg.OLIVA, "-"), (0.98, 4.0, eg.LADRILLO, "--")]:
    g0 = s2z / (1 - phi**2)
    vr = (q * s2w + 2 * g0 * (1 - phi**q)) / (q * (s2w + 2 * g0 * (1 - phi)))
    lim = s2w / (s2w + 2 * g0 * (1 - phi))
    ax[1].plot(q, vr, color=col, ls=ls, label=f"$\\phi={phi}$; límite {lim:.2f}".replace(".", ","))
    ax[1].axhline(lim, color=col, lw=0.6, ls=":")
ax[1].axhline(1, color="black", lw=0.6, ls=":")
ax[1].set_title("Paseo aleatorio + componente transitorio"); ax[1].set_xlabel("horizonte $q$"); ax[1].legend(fontsize=8)
for a in ax: eg.coma(a)
fig.savefig("graficos/cap07_fig_vr_teoria.pdf")

# --- Simulación: tamaño de las pruebas z y z* con rendimientos GARCH(1,1) ---
def vr_lm(r, q):
    nq = len(r); mu = r.mean(); sa = ((r - mu)**2).sum() / (nq - 1)
    rq = np.convolve(r, np.ones(q), "valid"); m = q * (nq - q + 1) * (1 - q / nq)
    vr = ((rq - q * mu)**2).sum() / m / sa
    z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * nq))
    e2 = (r - mu)**2; den = e2.sum()**2
    th = sum((2 * (q - j) / q)**2 * nq * (e2[j:] * e2[:-j]).sum() / den for j in range(1, q))
    return z, (vr - 1) / np.sqrt(th / nq)
rng = np.random.default_rng(77); T, reps = 1000, 2000; qs = [2, 5, 10, 20]
rech = {"z": np.zeros(len(qs)), "z*": np.zeros(len(qs))}
for _ in range(reps):
    e = rng.standard_normal(T + 200); h = np.ones(T + 200); r = np.zeros(T + 200)
    for t in range(1, T + 200):
        h[t] = 0.05 + 0.10 * r[t - 1]**2 + 0.85 * h[t - 1]; r[t] = np.sqrt(h[t]) * e[t]
    r = r[200:]
    for i, qq in enumerate(qs):
        z, zs = vr_lm(r, qq); rech["z"][i] += abs(z) > 1.96; rech["z*"][i] += abs(zs) > 1.96
for k in rech: rech[k] = 100 * rech[k] / reps
print("Rendimientos GARCH(1,1) sin autocorrelación, T = 1000, % de rechazos al 5 %:")
for k in rech: print(f"  {k:<3}", dict(zip(qs, np.round(rech[k], 1))))
fig, ax = plt.subplots(figsize=(5.8, 3.3)); w = 0.35; x = np.arange(len(qs))
ax.bar(x - w / 2, rech["z"], w, color=eg.LADRILLO, label="$z$ (supone homocedasticidad)")
ax.bar(x + w / 2, rech["z*"], w, color=eg.AZUL, label="$z^*$ (robusto)")
ax.axhline(5, color="black", ls=":", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([f"q = {v}" for v in qs])
ax.set_ylabel("% de rechazos de $H_0$ cierta"); ax.legend(fontsize=8); eg.coma(ax)
ax.xaxis.set_major_formatter(plt.FixedFormatter([f"q = {v}" for v in qs]))
fig.savefig("graficos/cap07_fig_vr_tamano.pdf")
