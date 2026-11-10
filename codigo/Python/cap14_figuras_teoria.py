# Capítulo 14: gráfica de la teoría (geometría de la prueba GRS en el plano media-desviación estándar)
import numpy as np
import estilo_graficos as eg
import matplotlib.pyplot as plt

def frontera(mu, V, objetivos):
    """Desviación estándar mínima para cada media objetivo (frontera sin activo libre de riesgo)."""
    i = np.ones(len(mu)); Vi = np.linalg.inv(V); A = i @ Vi @ i; B = i @ Vi @ mu; C = mu @ Vi @ mu; D = A * C - B**2
    return np.sqrt((A * objetivos**2 - 2 * B * objetivos + C) / D)
# Ejemplo ilustrativo: un factor (mercado) y dos activos de prueba con alfa positivo
mu_f = np.array([0.6]); Vf = np.array([[18.0]])
beta = np.array([1.1, 0.9]); alfa = np.array([0.25, 0.15]); Se = np.diag([6.0, 5.0])
mu = np.concatenate([mu_f, alfa + beta * mu_f]); V = np.block([[Vf, Vf @ beta[None, :]], [beta[:, None] @ Vf, np.outer(beta, beta) * Vf[0, 0] + Se]])
sr_f = mu_f[0] / np.sqrt(Vf[0, 0]); sr_all = np.sqrt(mu @ np.linalg.solve(V, mu)); a2 = alfa @ np.linalg.solve(Se, alfa)
print(f"SR del factor = {sr_f:.4f}; SR de la cartera tangente con todos = {sr_all:.4f}")
print(f"SR_todos^2 - SR_f^2 = {sr_all**2 - sr_f**2:.5f};  alfa' Sigma^-1 alfa = {a2:.5f}")
m = np.linspace(0, 2.2, 200); s_all = frontera(mu, V, m)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(s_all, m, color=eg.AZUL, lw=1.4, label="frontera con el factor y los activos de prueba")
ax.plot(np.sqrt(Vf[0, 0]), mu_f[0], "o", color=eg.LADRILLO, ms=6, label="factor (mercado)")
for i in range(2): ax.plot(np.sqrt(V[i + 1, i + 1]), mu[i + 1], "s", color=eg.GRIS, ms=5)
x = np.linspace(0, 9, 10)
ax.plot(x, sr_f * x, color=eg.LADRILLO, lw=1.2, ls="--", label=f"pendiente SR$_f$ = {sr_f:.3f}".replace(".", ","))
ax.plot(x, sr_all * x, color=eg.AZUL, lw=1.2, ls=":", label=f"pendiente SR$_{{\\mathrm{{todos}}}}$ = {sr_all:.3f}".replace(".", ","))
ax.set_xlim(0, 9); ax.set_ylim(0, 2.2); ax.set_xlabel("desviación estándar mensual (%)"); ax.set_ylabel("media mensual (%)")
ax.legend(fontsize=7.5, loc="upper left"); eg.coma(ax); fig.savefig("graficos/cap14_fig_grs.pdf")
