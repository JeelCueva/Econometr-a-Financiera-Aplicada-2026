# Capítulo 10, Ejercicio 10.5: ¿pronostica el VAR mejor que la media? Evaluación fuera de muestra (cartera pequeña)
import numpy as np, pandas as pd
from scipy import stats

f = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv")
y = 100 * f["S1V3"].to_numpy(); x = 100 * f["S5V3"].to_numpy(); T = len(y); fechas = f["dates"].to_numpy()
t0 = int(np.flatnonzero(fechas == "1983-01")[0])          # primer mes pronosticado (índice desde 0)
pm, pa, pv = [], [], []
for t in range(t0, T):                                     # ventana creciente: datos hasta t-1
    yy, y1, x1 = y[1:t], y[:t - 1], x[:t - 1]
    pm.append(y[:t].mean())
    ba = np.linalg.lstsq(np.column_stack([np.ones(t - 1), y1]), yy, rcond=None)[0]; pa.append(ba[0] + ba[1] * y[t - 1])
    bv = np.linalg.lstsq(np.column_stack([np.ones(t - 1), y1, x1]), yy, rcond=None)[0]
    pv.append(bv[0] + bv[1] * y[t - 1] + bv[2] * x[t - 1])
pm, pa, pv = map(np.array, (pm, pa, pv)); real = y[t0:]; em, ea, ev = real - pm, real - pa, real - pv
print(f"Pronósticos a un mes: {fechas[t0]} a {fechas[-1]} ({len(real)} meses); estimación inicial {fechas[0]} a {fechas[t0 - 1]}")
r2 = lambda e: 100 * (1 - (e**2).sum() / (em**2).sum())
print(f"ECM: media = {(em**2).mean():.4f}; AR(1) = {(ea**2).mean():.4f}; VAR(1) = {(ev**2).mean():.4f}")
print(f"R2 fuera de muestra respecto de la media: AR(1) = {r2(ea):.2f} %; VAR(1) = {r2(ev):.2f} %")

def clark_west(e1, e2, p1, p2):                            # modelo 1 anidado en el 2; H0: igual precisión
    fcw = e1**2 - (e2**2 - (p1 - p2)**2); n = len(fcw); u = fcw - fcw.mean()
    L = int(np.floor(4 * (n / 100)**(2 / 9))); s = (u**2).sum() / n
    for j in range(1, L + 1): s += 2 * (1 - j / (L + 1)) * (u[j:] * u[:-j]).sum() / n
    t = fcw.mean() / np.sqrt(s / n); return t, stats.norm.sf(t)
for et, args in [("media vs AR(1)", (em, ea, pm, pa)), ("media vs VAR(1)", (em, ev, pm, pv)), ("AR(1) vs VAR(1)", (ea, ev, pa, pv))]:
    t, p = clark_west(*args)
    print(f"{'Clark-West (unilateral):' if et.startswith('media vs AR') else '':<25}{et}: t = {t:.2f} (p = {p:.4f})")
for a, b in [("1983-01", "1999-12"), ("2000-01", "2017-12")]:
    k = (fechas[t0:] >= a) & (fechas[t0:] <= b)
    print(f"  {a} a {b}: R2 fuera de muestra AR(1) = {100 * (1 - (ea[k]**2).sum() / (em[k]**2).sum()):6.2f} %; "
          f"VAR(1) = {100 * (1 - (ev[k]**2).sum() / (em[k]**2).sum()):6.2f} %")

# ---- Figura: diferencia acumulada de errores cuadráticos (media menos modelo) ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fe = pd.to_datetime(fechas[t0:])
fig, ax = plt.subplots(figsize=(8.6, 3.4))
ax.plot(fe, np.cumsum(em**2 - ea**2), color=eg.OLIVA, lw=1.1, ls="--", label="AR(1)")
ax.plot(fe, np.cumsum(em**2 - ev**2), color=eg.LADRILLO, lw=1.3, label="VAR(1): pequeña y grande")
ax.axhline(0, color="black", lw=0.6)
ax.set_ylabel("$\\sum (e^2_{\\mathrm{media}} - e^2_{\\mathrm{modelo}})$"); ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
fig.savefig("graficos/cap10_ej05_fuera_muestra.pdf")
