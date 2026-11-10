# Capítulo 14, Ejercicio 14.2: la prueba GRS con tres conjuntos de carteras y tres modelos
from cap14_funciones import datos, grs, VALOR, MOMENTUM, INDUSTRIAS

d, R, F = datos()
conj = {"tamaño y valor (9)": VALOR, "tamaño y momentum (9)": MOMENTUM, "industrias (12)": INDUSTRIAS}
mods = {"CAPM": ["MktRF"], "FF3": ["MktRF", "SMB", "HML"], "Carhart": ["MktRF", "SMB", "HML", "Mom"]}
print("Carteras                 Modelo     GRS       gl         valor p     |alfa| medio   SR2 factores")
res = {}
for cn, cols in conj.items():
    for mn, fac in mods.items():
        g = grs(R[cols], F[fac]); res[(cn, mn)] = g
        print(f"{cn:<24} {mn:<8} {g['F']:7.3f}  ({g['gl'][0]:2d}, {g['gl'][1]:3d})  {g['p']:10.2e}   {g['alfa_abs']:8.3f}      {g['sr2_f']:8.4f}")

# ---- Figura: alfas de las carteras de tamaño y momentum según el modelo ----
import numpy as np, estilo_graficos as eg, matplotlib.pyplot as plt
from cap14_funciones import regs_ts
x = np.arange(9); w = 0.27
fig, ax = plt.subplots(figsize=(8.6, 3.4))
for j, (mn, col) in enumerate(zip(mods, [eg.GRIS, eg.AZUL, eg.LADRILLO])):
    ax.bar(x + (j - 1) * w, regs_ts(R[MOMENTUM], F[mods[mn]])["alfa"], w, color=col, label=mn)
ax.axhline(0, color="black", lw=0.5); ax.set_xticks(x); ax.set_xticklabels(MOMENTUM)
ax.set_ylabel("alfa (% mensual)"); ax.legend(fontsize=8); eg.coma(ax); ax.set_xticklabels(MOMENTUM)
fig.savefig("graficos/cap14_ej02_alfas_momentum.pdf")
