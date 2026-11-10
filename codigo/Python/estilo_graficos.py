"""Estilo común de las gráficas del libro (paleta sobria y tipografía Libertinus si está disponible)."""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

AZUL, LADRILLO, OLIVA, GRIS = "#1F3C6E", "#A63E2E", "#687838", "#787878"

for f in glob.glob("/usr/share/texlive/texmf-dist/fonts/opentype/public/libertinus-fonts/LibertinusSerif-*.otf"):
    font_manager.fontManager.addfont(f)
familia = "Libertinus Serif" if any("Libertinus Serif" in f.name for f in font_manager.fontManager.ttflist) else "serif"

plt.rcParams.update({
    "font.family": familia, "mathtext.fontset": "stix", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
    "axes.formatter.use_locale": False, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight",
})

def coma(ax):
    """Usa coma decimal en las marcas de los ejes (convención peruana)."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(f); ax.yaxis.set_major_formatter(f)
