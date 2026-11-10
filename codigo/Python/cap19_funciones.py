# Capítulo 19: funciones propias para datos de panel
import numpy as np, pandas as pd

def grunfeld():
    d = pd.read_csv("datos/internacional/grunfeld_inversion.csv")
    return d["empresa"].to_numpy(), d["anio"].to_numpy(), d["inversion"].to_numpy(), d[["valor", "capital"]].to_numpy()

def mco(y, X):
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; return b, y - X @ b, XtXi

def media_grupo(z, g):
    """Media de z dentro de cada grupo, repetida en cada observación."""
    u, inv = np.unique(g, return_inverse=True); z = np.asarray(z, float)
    if z.ndim == 1: return (np.bincount(inv, z) / np.bincount(inv))[inv]
    return np.column_stack([media_grupo(z[:, j], g) for j in range(z.shape[1])])

def v_clasica(X, e, XtXi, gl): return (e @ e) / gl * XtXi

def v_white(X, e, XtXi, k):
    n = len(e); return n / (n - k) * XtXi @ (X.T * e**2) @ X @ XtXi

def v_cluster(X, e, XtXi, g, k):
    """Errores agrupados por g con el ajuste de muestra finita de Stata: G/(G-1) (n-1)/(n-k)."""
    n = len(e); grupos = np.unique(g); G = len(grupos); S = np.zeros((X.shape[1], X.shape[1]))
    for c in grupos:
        s = X[g == c].T @ e[g == c]; S += np.outer(s, s)
    return G / (G - 1) * (n - 1) / (n - k) * XtXi @ S @ XtXi

def v_dos_vias(X, e, XtXi, g1, g2, k):
    """Agrupamiento en dos dimensiones (Cameron, Gelbach y Miller, 2011; Thompson, 2011)."""
    return v_cluster(X, e, XtXi, g1, k) + v_cluster(X, e, XtXi, g2, k) - v_white(X, e, XtXi, k)

def v_driscoll_kraay(X, e, XtXi, t, L):
    """Driscoll y Kraay (1998): Newey-West (Bartlett, L rezagos) sobre h_t = sum_i x_it e_it."""
    ts = np.unique(t); H = np.array([X[t == s].T @ e[t == s] for s in ts]); S = H.T @ H
    for j in range(1, L + 1): G = H[j:].T @ H[:-j]; S += (1 - j / (L + 1)) * (G + G.T)
    return XtXi @ S @ XtXi

def efectos_aleatorios(y, X, g):
    """Efectos aleatorios de Swamy y Arora para un panel balanceado. X sin constante."""
    N = len(np.unique(g)); n, K = X.shape; T = n // N
    yw, Xw = y - media_grupo(y, g), X - media_grupo(X, g); bw, ew, _ = mco(yw, Xw)
    s2e = (ew @ ew) / (n - N - K)
    _, idx = np.unique(g, return_index=True); yb, Xb = media_grupo(y, g)[idx], media_grupo(X, g)[idx]
    Zb = np.column_stack([np.ones(N), Xb]); _, eb, _ = mco(yb, Zb); s21 = T * (eb @ eb) / (N - K - 1)
    th = 1 - np.sqrt(s2e / s21)
    ys = y - th * media_grupo(y, g); Xs = np.column_stack([np.full(n, 1 - th), X - th * media_grupo(X, g)])
    b, e, XtXi = mco(ys, Xs); V = (e @ e) / (n - K - 1) * XtXi
    return b, V, th, s2e, (s21 - s2e) / T

def mc2e(y, X, Z):
    """Mínimos cuadrados en dos etapas (variables instrumentales) con errores clásicos."""
    PZ = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X); b = np.linalg.solve(PZ.T @ X, PZ.T @ y); e = y - X @ b
    V = (e @ e) / (len(y) - X.shape[1]) * np.linalg.inv(PZ.T @ PZ); return b, V

def rezagar(z, g, t, k=1):
    """Valor de z en t - k para la misma unidad (NaN si no existe)."""
    clave = {(a, b): v for a, b, v in zip(g, t, z)}
    return np.array([clave.get((a, b - k), np.nan) for a, b in zip(g, t)])

class LCG:
    M = 2147483647
    def __init__(self, s): self.x = s
    def u(self): self.x = (16807 * self.x) % self.M; return self.x / self.M
    def normal(self):
        u1, u2 = self.u(), self.u(); return np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)

CARTERAS = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other",
            "S1V1", "S1V3", "S1V5", "S3V1", "S3V3", "S3V5", "S5V1", "S5V3", "S5V5",
            "S1M1", "S1M3", "S1M5", "S3M1", "S3M3", "S3M5", "S5M1", "S5M3", "S5M5"]

def panel_momentum():
    """Panel de 30 carteras de French: exceso de rendimiento (%) del mes t y rendimiento acumulado de t-12 a t-2 (%)."""
    d = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv"); R = 100 * (d[CARTERAS].to_numpy() - d[["RF"]].to_numpy())
    Rb = 100 * d[CARTERAS].to_numpy(); T, N = R.shape
    S = np.full((T, N), np.nan)
    for k in range(12, T): S[k] = Rb[k - 12:k - 1].sum(0)
    f = d["dates"].astype(str).to_numpy(); k = np.arange(12, T)
    y = R[k].ravel(); x = S[k].ravel(); cart = np.tile(np.arange(N), len(k)); mes = np.repeat(k, N)
    return y, x, cart, mes, f
