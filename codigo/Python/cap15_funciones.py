# Capítulo 15: funciones propias para la estructura temporal de tasas de interés
import numpy as np, pandas as pd

TAU = np.array([1, 2, 3, 5, 6, 11, 12, 36, 60, 120])     # vencimientos en meses

def cargar():
    d = pd.read_csv("datos/internacional/tasas_cupon_cero_eeuu_mk.csv")
    return d["fecha"].to_numpy(), d[[f"r{m}" for m in TAU]].to_numpy()

def ns_cargas(tau, lam):
    """Cargas de Nelson-Siegel: nivel, pendiente y curvatura."""
    x = lam * np.asarray(tau, float); s = (1 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(x), s, s - np.exp(-x)])

def ns_ajuste(Y, tau, lam):
    """MCO de corte transversal, mes a mes, con lambda fijo: factores, curva ajustada y RECM."""
    X = ns_cargas(tau, lam); B = np.linalg.solve(X.T @ X, X.T @ Y.T).T; A = B @ X.T
    return B, A, np.sqrt(((Y - A)**2).mean())

def ns_lambda(Y, tau, malla):
    """Lambda común que minimiza la suma de errores cuadráticos de todos los meses (búsqueda en malla)."""
    sse = np.array([((Y - ns_ajuste(Y, tau, l)[1])**2).sum() for l in malla])
    return malla[sse.argmin()], sse

def mco_nw(y, X, L):
    """MCO con constante y errores estándar de Newey-West (núcleo de Bartlett, L rezagos)."""
    X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.solve(X.T @ X, X.T @ y); e = y - X @ b; n = len(y)
    g = X * e[:, None]; S = g.T @ g
    for j in range(1, L + 1): G = g[j:].T @ g[:-j]; S += (1 - j / (L + 1)) * (G + G.T)
    Xi = np.linalg.inv(X.T @ X); V = Xi @ S @ Xi
    return b, np.sqrt(np.diag(V)), 1 - (e**2).sum() / ((y - y.mean())**2).sum(), n
