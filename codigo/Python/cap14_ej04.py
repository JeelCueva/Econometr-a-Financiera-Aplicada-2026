# Capítulo 14, Ejercicio 14.4: la prueba GRS como comparación de razones de Sharpe
import numpy as np
from cap14_funciones import datos, sr2, grs, VALOR

d, Rall, F = datos(); R = Rall[VALOR].to_numpy()
mods = {"CAPM": ["MktRF"], "FF3": ["MktRF", "SMB", "HML"], "Carhart": ["MktRF", "SMB", "HML", "Mom"]}
for mn, fac in mods.items():
    sf = sr2(F[fac]); sa = sr2(np.column_stack([F[fac], R])); g = grs(R, F[fac])
    print(f"{mn:<8} SR factores = {np.sqrt(sf):.4f} (anual {np.sqrt(12 * sf):.3f}); SR factores + 9 carteras = {np.sqrt(sa):.4f} (anual {np.sqrt(12 * sa):.3f})")
    print(f"         SR2 todos - SR2 factores = {sa - sf:.6f};  alfa' Sigma_MV^-1 alfa = {g['a2_mv']:.6f}")
s = F.mean() / F.std(ddof=1)
print(f"\nSR mensual de cada factor por separado: Mkt {s['MktRF']:.4f}, SMB {s['SMB']:.4f}, HML {s['HML']:.4f}, Mom {s['Mom']:.4f}")

# ---- Figura: razón de Sharpe anual máxima alcanzable con cada conjunto ----
import estilo_graficos as eg, matplotlib.pyplot as plt
conj = [("Mkt", F[["MktRF"]]), ("Mkt + 9", np.column_stack([F[["MktRF"]], R])), ("FF3", F[mods["FF3"]]),
        ("FF3 + 9", np.column_stack([F[mods["FF3"]], R])), ("Carhart", F[mods["Carhart"]]), ("Carhart + 9", np.column_stack([F[mods["Carhart"]], R]))]
v = [np.sqrt(12 * sr2(X)) for _, X in conj]
fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.bar(range(6), v, color=[eg.GRIS, eg.LADRILLO] * 3)
for k, x in enumerate(v): ax.text(k, x + 0.02, f"{x:.2f}".replace(".", ","), ha="center", fontsize=8)
ax.set_xticks(range(6)); ax.set_xticklabels([n for n, _ in conj]); ax.set_ylabel("razón de Sharpe anual máxima"); eg.coma(ax)
ax.set_xticklabels([n for n, _ in conj]); fig.savefig("graficos/cap14_ej04_sharpe.pdf")
