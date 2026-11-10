# Capítulo 13: funciones propias para VaR, ES, pruebas retrospectivas y teoría de valores extremos
import numpy as np
from scipy import stats
from scipy.optimize import minimize

def var_es_normal(mu, s, p):
    """Pérdida L = -r con r ~ N(mu, s^2): VaR y ES al nivel p."""
    z = stats.norm.ppf(p); return -mu + s * z, -mu + s * stats.norm.pdf(z) / (1 - p)

def var_es_t(mu, s, nu, p):
    """r = mu + s z con z t de Student estandarizada (varianza 1)."""
    c = np.sqrt((nu - 2) / nu); q = stats.t.ppf(p, nu)
    es_t = stats.t.pdf(q, nu) / (1 - p) * (nu + q**2) / (nu - 1)   # E[X | X > q] para X ~ t_nu
    return -mu + s * c * q, -mu + s * c * es_t

def var_es_hist(L, p):
    """Simulación histórica: cuantil empírico (interpolación lineal, como type = 7 en R) y media de la cola."""
    v = np.quantile(L, p); return v, L[L >= v].mean()

def kupiec(x, p):
    """Prueba de cobertura incondicional: x = 1 si la pérdida superó al VaR."""
    n, k = len(x), int(x.sum()); pi = k / n; q = 1 - p
    l0 = (n - k) * np.log(1 - q) + k * np.log(q)
    l1 = (n - k) * np.log(1 - pi) + (k * np.log(pi) if k > 0 else 0.0)
    LR = -2 * (l0 - l1); return LR, stats.chi2.sf(LR, 1)

def christoffersen(x):
    """Prueba de independencia de las excepciones (cadena de Markov de primer orden)."""
    a, b = x[:-1], x[1:]
    n00, n01 = ((a == 0) & (b == 0)).sum(), ((a == 0) & (b == 1)).sum()
    n10, n11 = ((a == 1) & (b == 0)).sum(), ((a == 1) & (b == 1)).sum()
    p01, p11, p1 = n01 / (n00 + n01), n11 / (n10 + n11) if n10 + n11 > 0 else 0.0, (n01 + n11) / (n00 + n01 + n10 + n11)
    xl = lambda n, q: n * np.log(q) if n > 0 else 0.0
    l0 = xl(n00 + n10, 1 - p1) + xl(n01 + n11, p1)
    l1 = xl(n00, 1 - p01) + xl(n01, p01) + xl(n10, 1 - p11) + xl(n11, p11)
    LR = -2 * (l0 - l1); return LR, stats.chi2.sf(LR, 1), (n00, n01, n10, n11)

def gpd_nll(par, y):
    """Menos log-verosimilitud de la Pareto generalizada para los excesos y > 0."""
    xi, beta = par
    if beta <= 0: return 1e10
    if abs(xi) < 1e-8: return len(y) * np.log(beta) + y.sum() / beta
    w = 1 + xi * y / beta
    if np.any(w <= 0): return 1e10
    return len(y) * np.log(beta) + (1 + 1 / xi) * np.log(w).sum()

def gpd_fit(y):
    o = minimize(gpd_nll, [0.1, y.mean()], args=(y,), method="Nelder-Mead",
                 options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    xi, beta = o.x; h = 1e-5 * np.maximum(np.abs(o.x), 1); E = np.eye(2) * h
    f = lambda q: gpd_nll(q, y)
    H = np.array([[(f(o.x + E[i] + E[j]) - f(o.x + E[i] - E[j]) - f(o.x - E[i] + E[j]) + f(o.x - E[i] - E[j]))
                   / (4 * h[i] * h[j]) for j in range(2)] for i in range(2)])
    return xi, beta, np.sqrt(np.diag(np.linalg.inv(H))), -o.fun

def gpd_var_es(u, xi, beta, n, nu, p):
    """VaR y ES con la cola Pareto generalizada (n observaciones, nu excesos sobre u)."""
    v = u + beta / xi * (((n / nu) * (1 - p))**(-xi) - 1)
    return v, v / (1 - xi) + (beta - xi * u) / (1 - xi)
