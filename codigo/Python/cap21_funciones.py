# Capítulo 21: funciones del proyecto integrador (reutiliza los Capítulos 8 y 13)
import numpy as np, pandas as pd
from scipy import stats
from cap08_funciones import garch_fit, garch_s2, backcast
from cap13_funciones import var_es_t, var_es_normal, var_es_hist, kupiec, christoffersen

NOMBRES = ["AAPL", "AMZN", "FB", "GOOG", "SP500"]

def cargar():
    """Rendimientos simples diarios (%) de las cuatro acciones y del S&P 500, 2014-2018."""
    d = pd.read_csv("datos/internacional/gafa_sp500_diario_2014_2018.csv"); P = d[NOMBRES].to_numpy()
    return d["fecha"].to_numpy()[1:], 100 * (P[1:] / P[:-1] - 1)

def asimetria(x): c = x - x.mean(); return (c**3).mean() / (c**2).mean()**1.5
def curtosis(x): c = x - x.mean(); return (c**4).mean() / (c**2).mean()**2

def ljung_box(x, m):
    n = len(x); c = x - x.mean(); g0 = c @ c
    r = np.array([c[k:] @ c[:-k] / g0 for k in range(1, m + 1)]); Q = n * (n + 2) * np.sum(r**2 / (n - np.arange(1, m + 1)))
    return Q, stats.chi2.sf(Q, m)

def nw_mco(y, X, L):
    """MCO con errores de Newey-West (Bartlett, L rezagos). X incluye la constante."""
    b = np.linalg.solve(X.T @ X, X.T @ y); e = y - X @ b; g = X * e[:, None]; S = g.T @ g
    for j in range(1, L + 1): G = g[j:].T @ g[:-j]; S += (1 - j / (L + 1)) * (G + G.T)
    A = np.linalg.inv(X.T @ X); return b, np.sqrt(np.diag(A @ S @ A)), 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))

def ewma_var(x, lam=0.94, n0=30):
    h = np.empty(len(x)); h[0] = (x[:n0]**2).mean()
    for i in range(1, len(x)): h[i] = lam * h[i - 1] + (1 - lam) * x[i - 1]**2
    return h

def min_varianza(S):
    """Pesos de la cartera de mínima varianza global: Sigma^{-1} 1 / (1' Sigma^{-1} 1)."""
    w = np.linalg.solve(S, np.ones(len(S))); return w / w.sum()

def max_caida(r):
    """Máxima caída (en %) de la riqueza acumulada con rendimientos simples r en %."""
    v = np.cumprod(1 + r / 100); return 100 * (1 - (v / np.maximum.accumulate(v)).min())
