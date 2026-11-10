# Capítulo 1, Ejercicio 1.3: rendimiento en soles de Southern Copper (SCCO) con dos fuentes del tipo de cambio
import numpy as np, pandas as pd

lee = lambda f, c: pd.read_csv(f"datos/peru/{f}", parse_dates=["fecha"], index_col="fecha")[c]
P = lee("scco_yahoo.csv", "cierre_ajustado")          # dólares, con dividendos y desdoblamientos
S_bcrp = lee("tc_interbancario_venta_bcrp.csv", "tc")  # BCRP PD04638PD, soles por dólar
S_yah = lee("pen_x_yahoo_mensual.csv", "tc")           # Yahoo PEN=X, último dato de cada mes

# Fechas de corte: fin de cada año 2014-2025 y el último fin de mes disponible (30-oct-2026)
cortes = pd.to_datetime([f"{a}-12-31" for a in range(2014, 2026)] + ["2026-10-30"])
ultimo = lambda x, d: x[x.index <= d].iloc[-1]          # último dato disponible hasta la fecha d
tab = pd.DataFrame({"P": [ultimo(P, d) for d in cortes],
                    "S_bcrp": [ultimo(S_bcrp, d) for d in cortes],
                    "S_yahoo": [ultimo(S_yah, d) for d in cortes]},
                   index=[d.year for d in cortes[:-1]] + ["2026*"])
lr = np.log(tab).diff().dropna()                        # rendimientos logarítmicos anuales
out = pd.DataFrame({"r_usd": lr["P"], "dlnS_bcrp": lr["S_bcrp"], "r_sol_bcrp": lr["P"] + lr["S_bcrp"],
                    "dlnS_yahoo": lr["S_yahoo"], "r_sol_yahoo": lr["P"] + lr["S_yahoo"]})
print("Descomposición anual (%, logarítmica):  r_soles = r_dólares + dln S")
print((100 * out).round(2).to_string())

# Verificación de la Proposición 1.16 en forma simple
V = tab["P"] * tab["S_bcrp"]
R_sol = V.pct_change().dropna(); R_usd = tab["P"].pct_change().dropna(); R_S = tab["S_bcrp"].pct_change().dropna()
print(f"\nMáx |R_sol - (R_usd + R_S + R_usd*R_S)| = {(R_sol - (R_usd + R_S + R_usd*R_S)).abs().max():.1e}")
print(f"Diferencia entre fuentes en dln S (pp): media = {100*(out.dlnS_bcrp-out.dlnS_yahoo).mean():.3f}, "
      f"máx abs = {100*(out.dlnS_bcrp-out.dlnS_yahoo).abs().max():.3f}")
tot = out.sum()
print(f"\nPeríodo completo (dic-2014 a oct-2026), suma de logaritmos (%):")
print(f"   acción en dólares {100*tot.r_usd:7.2f} | tipo de cambio BCRP {100*tot.dlnS_bcrp:6.2f}"
      f" | en soles {100*tot.r_sol_bcrp:7.2f}")
print(f"   rendimiento simple en soles: {100*(np.exp(tot.r_sol_bcrp)-1):.2f} %  (en dólares: {100*(np.exp(tot.r_usd)-1):.2f} %)")

# ---- Figura: barras apiladas por año ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 3.8))
x = np.arange(len(out)); y1 = 100 * out.r_usd; y2 = 100 * out.dlnS_bcrp
ax.bar(x, y1, color=eg.AZUL, width=0.65, label="acción en dólares $r^{\\$}_t$")
ax.bar(x, y2, bottom=np.where(np.sign(y2) == np.sign(y1), y1, 0), color=eg.LADRILLO, width=0.65,
       label="tipo de cambio $\\Delta\\ln S_t$ (BCRP)")
ax.plot(x, 100 * out.r_sol_bcrp, "D", color="black", ms=4, label="total en soles")
ax.axhline(0, color=eg.GRIS, lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(out.index.astype(str)); ax.set_ylabel("% (logarítmico)")
ax.legend(fontsize=8, loc="upper left"); eg.coma(ax); ax.set_xticklabels(out.index.astype(str))
fig.savefig("graficos/cap01_fig_soles.pdf")
