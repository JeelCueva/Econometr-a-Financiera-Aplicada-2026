# Capítulo 18, Ejercicio 18.5: ¿cambia la dependencia en las crisis? Crisis financiera de 2007-2009 frente al resto
import numpy as np
from scipy.stats import kendalltau
from cap18_funciones import indices, pseudo, ewma_estandarizar, ajustar, tau_lambda, colas_empiricas

f, x, y = indices(); zx, zy = ewma_estandarizar(x), ewma_estandarizar(y)
crisis = (f >= "2007-07-01") & (f <= "2009-06-30")
print(f"Crisis: {f[crisis][0]} a {f[crisis][-1]} ({crisis.sum()} días); resto: {(~crisis).sum()} días")
print("\nmuestra   Pearson   tau      lambda_inf(5 %)  lambda_sup(5 %)")
for nom, k in [("crisis", crisis), ("resto ", ~crisis)]:
    u, v = pseudo(zx[k]), pseudo(zy[k]); li, ls = colas_empiricas(u, v, 0.05)
    print(f"{nom}   {np.corrcoef(zx[k], zy[k])[0, 1]:.4f}  {kendalltau(zx[k], zy[k])[0]:.4f}   {li:10.4f}      {ls:10.4f}")
print("\nmuestra  cópula      parámetro(s)               log-verosim.  lambda_inf lambda_sup")
for nom, k in [("crisis", crisis), ("resto ", ~crisis)]:
    A = ajustar(pseudo(zx[k]), pseudo(zy[k]))
    for c in ("gaussiana", "t", "clayton", "gumbel"):
        p, ll = A[c]; tl = tau_lambda(c, p)
        ps = f"rho = {p[0]:.4f}, nu = {p[1]:.3f}" if c == "t" else f"{'rho' if c == 'gaussiana' else 'theta'} = {p:.4f}"
        print(f"{nom}   {c:10s}  {ps:24s}  {ll:12.4f}  {tl[1]:8.4f}  {tl[2]:8.4f}")
# tau de Kendall móvil (250 días, cada 21 días)
ini = np.arange(0, len(x) - 250 + 1, 21); tm = np.array([kendalltau(zx[i:i + 250], zy[i:i + 250])[0] for i in ini])
print(f"\ntau de Kendall móvil (ventanas de 250 días cada 21 días, {len(ini)} ventanas): mínimo = {tm.min():.4f} (termina {f[ini[tm.argmin()] + 249]}); "
      f"máximo = {tm.max():.4f} (termina {f[ini[tm.argmax()] + 249]}); mediana = {np.median(tm):.4f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fin = ini + 249; pos = [k for k in range(len(f)) if f[k][5:7] == "01" and f[k][8:10] <= "07" and (k == 0 or f[k - 1][:4] != f[k][:4]) and int(f[k][:4]) % 3 == 0]
fig, ax = plt.subplots(figsize=(8.6, 3.0))
ax.axvspan(np.where(crisis)[0][0], np.where(crisis)[0][-1], color=eg.GRIS, alpha=0.3, lw=0, label="crisis 2007-2009")
ax.plot(fin, tm, color=eg.AZUL, lw=1.2, label="$\\tau$ de Kendall, 250 días")
ax.set_xticks(pos); ax.set_xticklabels([f[k][:4] for k in pos]); ax.set_ylabel("$\\tau$"); ax.legend(fontsize=8, loc="lower right"); eg.coma(ax)
ax.set_xticklabels([f[k][:4] for k in pos]); fig.savefig("graficos/cap18_ej05_tau_movil.pdf")
