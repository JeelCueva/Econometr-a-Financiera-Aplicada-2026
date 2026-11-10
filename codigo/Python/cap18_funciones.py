# Capítulo 18: funciones propias para cópulas
import numpy as np, pandas as pd
from scipy.stats import norm, t as tdist, rankdata

def indices():
    """Rendimientos logarítmicos diarios (%) del S&P 500 y del Nasdaq, 1999-2018."""
    a = pd.read_csv("datos/internacional/sp500_diario_yahoo.csv"); b = pd.read_csv("datos/internacional/nasdaq_diario_yahoo.csv")
    pa, pb = a["Adj Close"].to_numpy(), b["Adj Close"].to_numpy()
    return a["Date"].to_numpy()[1:], 100 * np.diff(np.log(pa)), 100 * np.diff(np.log(pb))

def pseudo(x):
    """Pseudo-observaciones: rango / (n + 1), con rangos promedio en caso de empates."""
    return rankdata(x) / (len(x) + 1)

def ewma_estandarizar(x, lam=0.94, n0=30):
    """Residuos estandarizados por una varianza EWMA (RiskMetrics) iniciada con la varianza de las n0 primeras observaciones."""
    h = np.empty(len(x)); h[0] = (x[:n0]**2).mean()
    for i in range(1, len(x)): h[i] = lam * h[i - 1] + (1 - lam) * x[i - 1]**2
    return x / np.sqrt(h)

# ---- log-densidades de cópula ----
def ll_gauss(r, u, v):
    x, y = norm.ppf(u), norm.ppf(v)
    return np.sum(-0.5 * np.log(1 - r**2) - (r**2 * (x**2 + y**2) - 2 * r * x * y) / (2 * (1 - r**2)))

def ll_t(r, nu, u, v):
    x, y = tdist.ppf(u, nu), tdist.ppf(v, nu)
    q = (x**2 + y**2 - 2 * r * x * y) / (1 - r**2)
    from scipy.special import gammaln
    c = gammaln((nu + 2) / 2) + gammaln(nu / 2) - 2 * gammaln((nu + 1) / 2) - 0.5 * np.log(1 - r**2)
    return np.sum(c - (nu + 2) / 2 * np.log(1 + q / nu) + (nu + 1) / 2 * (np.log(1 + x**2 / nu) + np.log(1 + y**2 / nu)))

def ll_clayton(th, u, v):
    return np.sum(np.log(1 + th) - (th + 1) * (np.log(u) + np.log(v)) - (2 + 1 / th) * np.log(u**-th + v**-th - 1))

def ll_gumbel(th, u, v):
    x, y = -np.log(u), -np.log(v); s = x**th + y**th; A = s**(1 / th)
    return np.sum(-A + x + y + (th - 1) * (np.log(x) + np.log(y)) + (1 / th - 2) * np.log(s) + np.log(A + th - 1))

def dorada(f, a, b, tol=1e-9):
    """Máximo de f en [a, b] por búsqueda de la sección áurea."""
    g = (np.sqrt(5) - 1) / 2; c, d = b - g * (b - a), a + g * (b - a); fc, fd = f(c), f(d)
    while b - a > tol:
        if fc > fd: b, d, fd = d, c, fc; c = b - g * (b - a); fc = f(c)
        else: a, c, fc = c, d, fd; d = a + g * (b - a); fd = f(d)
    x = (a + b) / 2; return x, f(x)

def ajustar(u, v):
    """Máxima verosimilitud canónica de las cuatro cópulas. Devuelve un diccionario con parámetros y log-verosimilitud."""
    out = {}
    out["gaussiana"] = dorada(lambda r: ll_gauss(r, u, v), -0.99, 0.99)
    prof = lambda nu: dorada(lambda r: ll_t(r, nu, u, v), -0.99, 0.99)[1]
    nus = np.arange(2, 31); best = nus[np.argmax([prof(n) for n in nus])]
    nu, _ = dorada(prof, max(2.0, best - 1.0), best + 1.0, tol=1e-6)
    r, ll = dorada(lambda r: ll_t(r, nu, u, v), -0.99, 0.99); out["t"] = ((r, nu), ll)
    out["clayton"] = dorada(lambda th: ll_clayton(th, u, v), 0.01, 20)
    out["gumbel"] = dorada(lambda th: ll_gumbel(th, u, v), 1.0001, 20)
    return out

# ---- tau de Kendall y dependencia en las colas implícitas ----
def tau_lambda(nombre, p):
    if nombre == "gaussiana": return 2 / np.pi * np.arcsin(p), 0.0, 0.0
    if nombre == "t":
        r, nu = p; l = 2 * tdist.cdf(-np.sqrt((nu + 1) * (1 - r) / (1 + r)), nu + 1); return 2 / np.pi * np.arcsin(r), l, l
    if nombre == "clayton": return p / (p + 2), 2**(-1 / p), 0.0
    if nombre == "gumbel": return 1 - 1 / p, 0.0, 2 - 2**(1 / p)

def colas_empiricas(u, v, q):
    """Estimadores empíricos de la dependencia en la cola inferior y superior al nivel q."""
    n = len(u); return ((u <= q) & (v <= q)).sum() / (n * q), ((u > 1 - q) & (v > 1 - q)).sum() / (n * q)

class LCG:
    """Generador de Park y Miller (1988), el mismo del Capítulo 16."""
    M = 2147483647
    def __init__(self, semilla): self.x = semilla
    def u(self, n=None):
        if n is None:
            self.x = (16807 * self.x) % self.M; return self.x / self.M
        return np.array([self.u() for _ in range(n)])

def phi2_diag(a, r, m=20001):
    """Pr(X <= a, Y <= a) para una normal bivariada estándar con correlación r (regla de Simpson en [-10, a])."""
    x = np.linspace(-10, a, m); f = norm.pdf(x) * norm.cdf((a - r * x) / np.sqrt(1 - r**2)); h = x[1] - x[0]
    return h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())
