# Capítulo 10, Ejercicio 10.2: causalidad de Granger (S&P 500 y VIX diarios; carteras pequeña y grande mensuales)
import numpy as np, pandas as pd
from cap10_funciones import granger, seleccion
from statsmodels.tsa.api import VAR                       # solo para comprobar

s = pd.read_csv("datos/internacional/sp500_diario_yahoo.csv"); v = pd.read_csv("datos/internacional/vix_diario.csv")
d = s[["Date", "Adj Close"]].merge(v[["DATE", "CLOSE"]], left_on="Date", right_on="DATE")
Y1 = np.column_stack([100 * np.diff(np.log(d["Adj Close"])), 100 * np.diff(np.log(d["CLOSE"]))])
f = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv"); Y2 = 100 * f[["S1V3", "S5V3"]].to_numpy()

def reporta(Y, p, efecto, causa, et):
    g = granger(Y, p, efecto, causa)
    print(f"  {et:<34} p = {p}: F({g['gl'][0]}, {g['gl'][1]}) = {g['F']:6.3f} (p = {g['pF']:.4f}); "
          f"Wald HC0 = {g['W']:6.3f} (p = {g['pW']:.4f})")
print("S&P 500 y VIX diarios, 1999-2018:")
for p in (2, 7): reporta(Y1, p, 0, [1], "dv no causa a r:"); reporta(Y1, p, 1, [0], "r no causa a dv:")
print("\nCarteras S1V3 (pequeña) y S5V3 (grande), mensuales 1949-2017:")
ic = seleccion(Y2, 6); print(f"  orden elegido: AIC = {int(ic[ic[:, 1].argmin(), 0])}, BIC = {int(ic[ic[:, 2].argmin(), 0])}")
for p in (1, 5): reporta(Y2, p, 0, [1], "la grande no causa a la pequeña:"); reporta(Y2, p, 1, [0], "la pequeña no causa a la grande:")
t = VAR(Y2).fit(1).test_causality(0, [1])
print(f"\ncomprobación con statsmodels (VAR(1), grande -> pequeña): F = {t.test_statistic:.3f}, gl = {t.df}, p = {t.pvalue:.4f}")
