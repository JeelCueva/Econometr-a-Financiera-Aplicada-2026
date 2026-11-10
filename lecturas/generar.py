"""Genera las tablas 'Cómo leer la salida' (lecturas/<id>.tex) a partir de lecturas/fuente/*.txt.
Formato: una línea '== <id>' abre una tabla; cada fila es 'Indicador | Valor | Interpretación'."""
import glob, re, os
CAB = (r"\par\medskip\noindent\textbf{Cómo leer la salida.}" "\n"
       r"{\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}" "\n"
       r"\begin{xltabular}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.24\linewidth}"
       r">{\raggedleft\arraybackslash}p{0.15\linewidth}>{\raggedright\arraybackslash}X@{}}" "\n"
       r"\toprule Indicador & Valor & Interpretación \\ \midrule \endfirsthead" "\n"
       r"\toprule Indicador & Valor & Interpretación \\ \midrule \endhead" "\n"
       r"\bottomrule \endlastfoot" "\n")
PIE = r"\end{xltabular}}" "\n"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
n = 0
for f in sorted(glob.glob("fuente/*.txt")):
    bloques = re.split(r"^== +(\S+)\s*$", open(f, encoding="utf-8").read(), flags=re.M)[1:]
    for ident, cuerpo in zip(bloques[::2], bloques[1::2]):
        filas = []
        for linea in cuerpo.strip().split("\n"):
            linea = linea.strip()
            if not linea or linea.startswith("#"): continue
            c = [x.strip() for x in linea.split(" | ")]
            assert len(c) == 3, (ident, linea)
            filas.append(" & ".join(c) + r" \\")
        open(f"{ident}.tex", "w", encoding="utf-8").write(CAB + "\n".join(filas) + "\n" + PIE)
        n += 1
print(n, "tablas")
