# Capítulo 21, Etapa 2: exposición al mercado (CAPM diario con errores de Newey-West)
import numpy as np
from cap21_funciones import cargar, NOMBRES, nw_mco

f, R = cargar(); n = len(f); m = R[:, 4]; X = np.column_stack([np.ones(n), m])
print(f"R_it = a_i + b_i R_mt + e_it con el S&P 500 como mercado; n = {n}; Newey-West con 5 rezagos")
print("acción   alfa anual (%)   t(alfa)    beta    t(beta = 1)    R2     vol. idiosincrática anual (%)")
B = []
for j, nom in enumerate(NOMBRES[:4]):
    b, ee, r2 = nw_mco(R[:, j], X, 5); e = R[:, j] - X @ b; B.append(b[1])
    print(f"{nom:6s}  {252 * b[0]:12.2f}  {b[0] / ee[0]:9.2f}  {b[1]:8.4f}  {(b[1] - 1) / ee[1]:10.2f}   {r2:7.4f}   {np.sqrt(252) * e.std(ddof=2):14.2f}")
w = np.full(4, 0.25); rp = R[:, :4] @ w; b, ee, r2 = nw_mco(rp, X, 5)
print(f"Cartera 1/N: alfa anual = {252 * b[0]:.2f} % (t = {b[0] / ee[0]:.2f}); beta = {b[1]:.4f}; R2 = {r2:.4f}; promedio de las betas = {np.mean(B):.4f}")
for a in ("2016-01-01",):
    k = f < a; b1, _, _ = nw_mco(rp[k], X[k], 5); b2, _, _ = nw_mco(rp[~k], X[~k], 5)
    print(f"Estabilidad: beta de la cartera 2014-2015 = {b1[1]:.4f}; 2016-2018 = {b2[1]:.4f}")
