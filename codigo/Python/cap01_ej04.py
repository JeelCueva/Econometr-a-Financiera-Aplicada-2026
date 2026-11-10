# Capítulo 1, Ejercicio 1.4: portafolio 1/3 BAP, SCCO y BVN; rendimiento exacto frente a suma de logaritmos
import numpy as np, pandas as pd

lee = lambda t: pd.read_csv(f"datos/peru/{t}_yahoo.csv", parse_dates=["fecha"], index_col="fecha")["cierre_ajustado"]
P = pd.DataFrame({t.upper(): lee(t) for t in ["bap", "scco", "bvn"]})
dup = pd.Timestamp("2020-04-08")                                  # ajuste duplicado de Yahoo en BAP
b = pd.read_csv("datos/peru/bap_yahoo.csv", parse_dates=["fecha"], index_col="fecha")
k = b["cierre_ajustado"] / b["cierre"]; salto = (k / k.shift(1))[dup]
P.loc[P.index < dup, "BAP"] *= salto                              # corrección del Ejercicio 1.1

Pm = P.groupby(P.index.to_period("M")).last()                     # último precio de cada mes
R = Pm.pct_change().dropna(); r = np.log(Pm).diff().dropna()
w = np.full(3, 1 / 3)
Rp = R @ w                                                        # Proposición 1.8 (exacto, rebalanceo mensual)
rp = np.log1p(Rp)                                                 # logaritmo exacto del portafolio
aprox = r @ w                                                     # aproximación sum w_i r_i
err = rp - aprox
disp = r.var(axis=1, ddof=0)                                      # dispersión transversal de r_i
print(f"Meses: {len(R)} ({R.index[0]} a {R.index[-1]})")
print(f"Error mínimo = {err.min():.2e} (Prop. 1.14 exige >= 0): {(err >= -1e-15).all()}")
print(f"Error medio = {10000*err.mean():.2f} pb, máximo = {10000*err.max():.2f} pb en {err.idxmax()}")
print(f"Media de disp/2 = {10000*(disp/2).mean():.2f} pb; correlación(error, disp/2) = {err.corr(disp/2):.4f}")
X = np.column_stack([np.ones(len(err)), disp]); beta = np.linalg.lstsq(X, err, rcond=None)[0]
print(f"Regresión error = a + b*disp:  a = {beta[0]:.2e}, b = {beta[1]:.4f}  (teoría de segundo orden: 0,5)")
print(f"\nSuma de errores en {len(R)} meses: {100*err.sum():.2f} puntos log")

V_reb = np.exp(rp.sum())                                          # 1 dólar invertido con rebalanceo
V_bh = (Pm.iloc[-1] / Pm.iloc[0]) @ w                             # comprar y mantener
print(f"\nValor final de 1 dólar (dic-2014 a {Pm.index[-1]}):")
print(f"   rebalanceo mensual: {V_reb:.4f}   comprar y mantener: {V_bh:.4f}")
print("   valor final por acción (comprar y mantener):", (Pm.iloc[-1] / Pm.iloc[0]).round(4).to_dict())
wf = (Pm.iloc[-1] / Pm.iloc[0]) * w / V_bh
print("   pesos finales sin rebalanceo:", wf.round(4).to_dict())

# ---- Figura: error de aproximación y media dispersión transversal ----
import estilo_graficos as eg
import matplotlib.pyplot as plt, matplotlib.dates as mdates
t = err.index.to_timestamp()
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.bar(t, 10000 * err, width=25, color=eg.AZUL, label="error $r_{p,t}-\\sum_i w_i r_{i,t}$")
ax.plot(t, 10000 * disp / 2, color=eg.LADRILLO, lw=0.9, label="$\\frac{1}{2}$ varianza transversal de $r_{i,t}$")
ax.set_ylabel("puntos básicos"); ax.legend(fontsize=8); eg.coma(ax)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.savefig("graficos/cap01_fig_portafolio.pdf")
