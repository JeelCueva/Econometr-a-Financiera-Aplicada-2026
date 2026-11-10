# Apéndice C: inventario de los archivos de datos/peru (número de filas, primera y última fecha y valor)
import glob, os, pandas as pd

for f in sorted(glob.glob("datos/peru/*.csv")):
    x = pd.read_csv(f)
    col = x.columns[1]
    print(f"{os.path.basename(f):34s} {len(x):5d}  {x.iloc[0, 0]} {x[col].iloc[0]:>10}  "
          f"{x.iloc[-1, 0]} {x[col].iloc[-1]:>10}")
