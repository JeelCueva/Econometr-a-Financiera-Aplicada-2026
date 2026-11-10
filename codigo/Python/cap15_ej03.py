# Capítulo 15, Ejercicio 15.3: rendimientos en exceso de mantener bonos y regresiones de Fama-Bliss
import numpy as np
from cap15_funciones import cargar, mco_nw, TAU

f, Y = cargar(); T = len(Y); col = {m: j for j, m in enumerate(TAU)}
print("Rendimiento en exceso a un mes del bono de n meses (en % mensual):")
print("  rx_{t+1}(n) = (n/12) y_{n,t} - ((n-1)/12) y_{n-1,t+1} - y_{1,t}/12")
print("Diferencial a plazo: (f_t(n) - y_{1,t})/12, con f_t(n) = n y_{n,t} - (n-1) y_{n-1,t}\n")
print("   n   media rx   desv.est.   Sharpe anual  |   b (Fama-Bliss)   ee     t(b=0)    R2")
RX = {}
for n in (2, 3, 6, 12):
    yn, yn1, y1 = Y[:, col[n]], Y[:, col[n - 1]], Y[:, col[1]]
    rx = (n / 12) * yn[:-1] - ((n - 1) / 12) * yn1[1:] - y1[:-1] / 12
    fs = (n * yn[:-1] - (n - 1) * yn1[:-1] - y1[:-1]) / 12
    b, ee, r2, _ = mco_nw(rx, fs, 1); RX[n] = (rx, fs, b)
    print(f"{n:4d} {rx.mean():9.4f} {rx.std(ddof=1):10.4f} {np.sqrt(12) * rx.mean() / rx.std(ddof=1):12.3f}   | {b[1]:12.4f} {ee[1]:10.4f} {b[1] / ee[1]:8.2f} {r2:8.4f}")

# ---- Figura: rendimiento en exceso frente al diferencial a plazo (n = 12) ----
import estilo_graficos as eg, matplotlib.pyplot as plt
rx, fs, b = RX[12]; fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(fs, rx, ".", color=eg.AZUL, ms=3, alpha=0.6); xs = np.linspace(fs.min(), fs.max(), 20)
ax.plot(xs, b[0] + b[1] * xs, color=eg.LADRILLO, lw=1.4, label=f"MCO: pendiente {b[1]:.2f}".replace(".", ","))
ax.plot(xs, np.full_like(xs, rx.mean()), color="black", ls="--", lw=0.9, label="expectativas: pendiente 0")
ax.set_xlabel("diferencial a plazo $(f_t(12)-y_{1,t})/12$ (% mensual)"); ax.set_ylabel("$rx_{t+1}(12)$ (% mensual)")
ax.legend(fontsize=8); eg.coma(ax); fig.savefig("graficos/cap15_ej03_fama_bliss.pdf")
