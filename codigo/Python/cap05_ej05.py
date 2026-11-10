# Capítulo 5, Ejercicio 5.5: ¿raíz unitaria o quiebre? Zivot-Andrews para la razón dividendo/precio
import numpy as np, pandas as pd
from cap05_funciones import adf
from arch.unitroot import ZivotAndrews                    # solo para comprobar

s = pd.read_csv("datos/internacional/sp500_mensual_shiller.csv", parse_dates=["Date"])
s = s[s.Date <= "2023-06-01"].reset_index(drop=True)
y = np.log((s.Dividend / s.SP500).to_numpy()); T = len(y)
a = adf(y, "ct"); p = a["p"]
print(f"ADF con tendencia: t = {a['t']:.3f} con {p} rezagos (VC 5% = -3,41)")
dy = np.diff(y); k = np.arange(p, len(dy))               # índices desde 0
Xb = np.column_stack([y[k]] + [dy[k - j] for j in range(1, p + 1)] + [np.ones(len(k)), k + 1.0])
corte = int(np.floor(0.15 * T))
cands = np.arange(corte + 1, T - corte + 1)               # TB desde 0: primer período del nuevo régimen
tz = []
for TB in cands:
    X = np.column_stack([Xb, (k + 1 >= TB).astype(float)])
    b, *_ = np.linalg.lstsq(X, dy[k], rcond=None); e = dy[k] - X @ b
    V = e @ e / (len(e) - X.shape[1]) * np.linalg.inv(X.T @ X); tz.append(b[0] / np.sqrt(V[0, 0]))
tz = np.array(tz); i = tz.argmin()
print(f"Zivot-Andrews: t mínimo = {tz[i]:.3f} en {s.Date[cands[i]].date()} (VC al 5%: -4,80)")
za = ZivotAndrews(y, lags=p, trend="c")
print(f"Comprobación con arch: {za.stat:.3f} (p-valor {za.pvalue:.3f}; VC 5% {za.critical_values['5%']:.2f})")

# ---- Figura ----
import estilo_graficos as eg
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
ax[0].plot(s.Date, y, color=eg.AZUL, lw=0.8); ax[0].axvline(s.Date[cands[i]], color=eg.LADRILLO, ls="--", lw=1)
ax[0].set_title("Logaritmo de la razón dividendo/precio")
ax[1].plot(s.Date[cands], tz, color=eg.AZUL, lw=0.8); ax[1].axhline(-4.80, color=eg.LADRILLO, ls="--", lw=1, label="valor crítico al 5 %")
ax[1].set_title("Estadístico de Zivot-Andrews según la fecha"); ax[1].legend(fontsize=8)
for a_ in ax: a_.yaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: f"{v_:g}".replace(".", ",")))
fig.savefig("graficos/cap05_fig_za.pdf")
