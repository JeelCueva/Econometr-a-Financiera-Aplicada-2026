# Capítulo 16, Ejercicio 16.5: varianza inducida por el evento (simulación) y el VIX en los anuncios del FOMC
import numpy as np, pandas as pd
from cap16_funciones import cargar, rendimientos_normales, pruebas, corrado, LCG

# (a) Simulación: el evento multiplica por 2,25 la varianza del día 0
f, R = cargar(); m = R[:, 4]; T = len(m); F0 = np.arange(250, T)
E, AR0, V0 = {}, {}, {}
for j in range(4):
    for t0 in F0:
        x = rendimientos_normales(R[:, j], m, t0, evt=(0, 0)); E[j, t0] = x["e"]; AR0[j, t0] = x["AR"][0]; V0[j, t0] = x["V"][0, 0]
N, REP = 20, 1000; g = LCG(20162)
sorteo = []
for _ in range(REP):
    s = []
    for _ in range(N):
        j = int(4 * g.u()); t0 = int(F0[0] + int(len(F0) * g.u())); u1, u2 = g.u(), g.u()
        s.append((j, t0, np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)))   # Box-Muller
    sorteo.append(s)
print("(a) Simulación: AR(0) + delta + ruido con varianza 1,25 V0 (varianza total x 2,25); rechazo si |est.| > 1,96")
print("delta (%)    J1     Patell    BMP    signo   Corrado")
for d in (0.0, 1.0):
    rech = np.zeros(5)
    for s in sorteo:
        a = np.array([AR0[j, t] + d + z * np.sqrt(1.25 * V0[j, t]) for j, t, z in s]); v = np.array([V0[j, t] for j, t, _ in s])
        p = pruebas(a, v, np.full(N, 238)); A = np.array([np.r_[E[j, t], a[i]] for i, (j, t, _) in enumerate(s)])
        rech += np.abs([p["J1"], p["Patell"], p["BMP"], p["signo"], corrado(A)]) > 1.96
    print(f"{d:6.1f}   " + "  ".join(f"{x:7.3f}" for x in rech / REP))

# (b) El VIX alrededor de los anuncios del FOMC: cambio logarítmico diario (%), modelo de media
v = pd.read_csv("datos/internacional/vix_diario.csv"); fv = v["DATE"].to_numpy()[1:]; x = 100 * np.diff(np.log(v["CLOSE"].to_numpy()))
ev = pd.read_csv("datos/internacional/fomc_anuncios_2014_2018.csv")["fecha_anuncio"].to_numpy()
T0 = np.array([int(np.where(fv == d)[0][0]) for d in ev]); res = [rendimientos_normales(x, x, t0, modelo="media") for t0 in T0]
AR = np.array([r["AR"] for r in res]); V = np.array([np.diag(r["V"]) for r in res]); gl = np.array([r["gl"] for r in res])
print(f"\n(b) Cambio porcentual del VIX, {len(T0)} anuncios del FOMC; estimación [-250, -11]")
print(" tau    CAAR      J1    Patell    BMP    signo  positivos  var(SAR)")
for k, tau in enumerate(range(-3, 4), start=2):
    p = pruebas(AR[:, k], V[:, k], gl)
    print(f" {tau:3d} {p['CAAR']:8.4f} {p['J1']:7.2f} {p['Patell']:7.2f} {p['BMP']:7.2f} {p['signo']:7.2f}   {p['pos']:2d}/{len(T0)}   {p['SAR'].var(ddof=1):7.4f}")
c = AR[:, 4:6].sum(1); vv = np.array([r["V"][4:6, 4:6].sum() for r in res]); sc = c / np.sqrt(vv)
print(f"CAAR(-1,0) = {c.mean():.4f}; J1 = {c.mean() / np.sqrt(vv.sum() / len(c)**2):.2f}; BMP = {sc.mean() / (sc.std(ddof=1) / np.sqrt(len(c))):.2f}")

# ---- Figura ----
import estilo_graficos as eg, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 3.2)); SAR0 = AR[:, 5] / np.sqrt(V[:, 5]); z = np.linspace(-4, 4, 200)
ax.hist(SAR0, bins=np.arange(-4, 4.01, 0.5), density=True, color=eg.AZUL, alpha=0.5, label="SAR del día 0 (VIX, 40 anuncios)")
ax.plot(z, np.exp(-z**2 / 2) / np.sqrt(2 * np.pi), color=eg.LADRILLO, lw=1.4, label="$N(0,1)$")
ax.set_xlabel("rendimiento anormal estandarizado"); ax.set_ylabel("densidad"); ax.legend(fontsize=8); eg.coma(ax)
fig.savefig("graficos/cap16_ej05_vix.pdf")
