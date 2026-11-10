# Capítulo 1, Ejercicio 1.1: dividendos de Credicorp (BAP) a partir del precio ajustado de Yahoo Finance
import numpy as np, pandas as pd

b = pd.read_csv("datos/peru/bap_yahoo.csv", parse_dates=["fecha"], index_col="fecha")
P, Pa = b["cierre"], b["cierre_ajustado"]
c = Pa / P                                    # c_t = P^a_t / P_t: constante entre eventos
salto = c / c.shift(1)                        # c_t / c_{t-1} = 1 / f_t
ev = salto[(salto - 1).abs() > 1e-4].index    # fechas ex-dividendo
Pprev = P.shift(1)
t = pd.DataFrame({"P_ant": Pprev[ev], "f": 1 / salto[ev]})
t["D"] = (1 - t["f"]) * t["P_ant"]            # D_t = (1 - c_{t-1}/c_t) P_{t-1}
t["d_%"] = 100 * t["D"] / t["P_ant"]
Rtot = (P[ev] + t["D"]) / t["P_ant"] - 1      # rendimiento total
Ra = (Pa / Pa.shift(1) - 1)[ev]               # rendimiento con precios ajustados
dd = t["D"] / t["P_ant"]
t["error_prop"] = (Ra - Rtot) - dd / (1 - dd) * Rtot
print(f"Observaciones: {len(P)}  ({P.index[0].date()} a {P.index[-1].date()})")
print(f"Eventos detectados: {len(ev)}\n")
print(t.assign(P_ant=t.P_ant.round(2), f=t.f.round(6), D=t.D.round(3),
               **{"d_%": t["d_%"].round(2)}).drop(columns="error_prop").to_string())
print(f"\nMáximo |error| en la Proposición 1.3: {t['error_prop'].abs().max():.2e}")

# Fuera de las fechas ex-dividendo, los rendimientos con P y con P^a coinciden
R, Ra_all = P.pct_change(), Pa.pct_change()
print(f"Máx |R^a - R| fuera de eventos: {(Ra_all - R).drop(ev).abs().max():.2e}")

# Rendimientos acumulados
acum = lambda x: x.iloc[-1] / x.iloc[0] - 1
dup = pd.Timestamp("2020-04-08")              # registro duplicado de Yahoo (un solo dividendo en 2020)
Pa_corr = Pa * np.where(Pa.index < dup, salto[dup], 1.0)
print(f"\nRendimiento acumulado con P (sin dividendos):       {100*acum(P):8.2f} %")
print(f"Rendimiento acumulado con P^a de Yahoo:             {100*acum(Pa):8.2f} %")
print(f"Rendimiento acumulado con P^a corregido (2020 x1):  {100*acum(Pa_corr):8.2f} %")
anios = (P.index[-1] - P.index[0]).days / 365.25
print(f"Tasa anual compuesta con P:             {100*((1+acum(P))**(1/anios)-1):6.2f} %")
print(f"Tasa anual compuesta con P^a corregido: {100*((1+acum(Pa_corr))**(1/anios)-1):6.2f} %")

# ---- Figura: precio de cierre y precio ajustado en escala logarítmica ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(P.index, P, color=eg.AZUL, lw=1, label="cierre $P_t$")
ax.plot(Pa.index, Pa, color=eg.LADRILLO, lw=1, label="ajustado de Yahoo $P^a_t$")
ax.plot(Pa_corr.index, Pa_corr, color=eg.OLIVA, lw=1, ls="--", label="ajustado corregido (un solo dividendo en 2020)")
for d in ev:
    ax.axvline(d, color=eg.GRIS, lw=0.4, alpha=0.6)
ax.set_yscale("log"); ax.set_ylabel("dólares (escala log)")
ax.set_yticks([60, 100, 150, 200, 300]); ax.set_yticklabels(["60", "100", "150", "200", "300"])
ax.minorticks_off(); ax.legend(loc="lower right", fontsize=8)
fig.savefig("graficos/cap01_fig_dividendos.pdf")
