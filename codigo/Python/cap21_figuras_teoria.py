# Capítulo 21: diagrama del flujo de trabajo del proyecto
import estilo_graficos as eg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

etapas = [("1. Datos y hechos\nestilizados", "Cap. 2-3, 6"), ("2. Exposición al\nmercado (CAPM)", "Cap. 4-5, 14"), ("3. Volatilidad\n(GARCH-t, EWMA)", "Cap. 8-9, 12"),
          ("4. VaR y ES con\nprueba retrospectiva", "Cap. 13"), ("5. Asignación\nfuera de muestra", "Cap. 12, 14, 20")]
fig, ax = plt.subplots(figsize=(10.0, 2.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 2.4); ax.axis("off")
for k, (t, c) in enumerate(etapas):
    x = 0.1 + 2.0 * k
    ax.add_patch(FancyBboxPatch((x, 0.9), 1.7, 1.1, boxstyle="round,pad=0.03", fc="white", ec=eg.AZUL, lw=1.2))
    ax.text(x + 0.85, 1.45, t, ha="center", va="center", fontsize=8.5); ax.text(x + 0.85, 0.6, c, ha="center", va="center", fontsize=7.5, color=eg.GRIS)
    if k < 4: ax.annotate("", xy=(x + 2.0, 1.45), xytext=(x + 1.75, 1.45), arrowprops=dict(arrowstyle="->", color="black", lw=1))
ax.annotate("", xy=(0.95, 0.25), xytext=(9.05, 0.25), arrowprops=dict(arrowstyle="->", color=eg.LADRILLO, lw=1, ls="--"))
ax.text(5.0, 0.08, "la evaluación fuera de muestra puede obligar a volver a una etapa anterior", ha="center", fontsize=7.5, color=eg.LADRILLO)
fig.savefig("graficos/cap21_fig_flujo.pdf")
