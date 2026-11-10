# Capítulo 10: funciones propias para estimar y analizar un VAR(p) por MCO
import numpy as np
from scipy import stats

def rezagos(Y, p, inicio=None):
    """Regresores [1, y_{t-1}', ..., y_{t-p}'] y regresando y_t; 'inicio' fija la primera fila usada."""
    Y = np.asarray(Y, float); T = len(Y); s = p if inicio is None else inicio
    X = np.column_stack([np.ones(T - s)] + [Y[s - j:T - j] for j in range(1, p + 1)])
    return Y[s:], X

def var_mco(Y, p, inicio=None):
    y, X = rezagos(Y, p, inicio); n, K = y.shape
    B = np.linalg.solve(X.T @ X, X.T @ y)                  # (1 + Kp) x K: una columna por ecuación
    U = y - X @ B
    return {"B": B, "U": U, "X": X, "y": y, "p": p, "K": K, "n": n,
            "S": U.T @ U / (n - X.shape[1]),                # Sigma con corrección de grados de libertad
            "S_mv": U.T @ U / n,                            # Sigma de máxima verosimilitud
            "A": [B[1 + K * (j - 1):1 + K * j].T for j in range(1, p + 1)]}   # A_j es K x K

def seleccion(Y, pmax):
    """Criterios de información con la misma muestra efectiva para todos los p."""
    filas = []
    for p in range(0, pmax + 1):
        if p == 0:
            y = np.asarray(Y, float)[pmax:]; U = y - y.mean(0); n, K = y.shape; S = U.T @ U / n
        else:
            m = var_mco(Y, p, inicio=pmax); n, K, S = m["n"], m["K"], m["S_mv"]
        ld = np.log(np.linalg.det(S)); k = p * K**2 + K
        filas.append((p, ld + 2 * k / n, ld + np.log(n) * k / n, ld + 2 * np.log(np.log(n)) * k / n))
    return np.array(filas)

def companera(A):
    K = A[0].shape[0]; p = len(A)
    F = np.zeros((K * p, K * p)); F[:K] = np.hstack(A)
    if p > 1: F[K:, :-K] = np.eye(K * (p - 1))
    return F

def irf(A, h, P=None):
    """Psi_0..Psi_h (P = None) o respuestas ortogonalizadas Theta_i = Psi_i P."""
    K = A[0].shape[0]; Psi = [np.eye(K)]
    for i in range(1, h + 1):
        Psi.append(sum(A[j - 1] @ Psi[i - j] for j in range(1, min(i, len(A)) + 1)))
    Psi = np.array(Psi)
    return Psi if P is None else Psi @ P

def fevd(Theta):
    """Proporción de la varianza del error de pronóstico de cada variable atribuida a cada choque."""
    acum = np.cumsum(Theta**2, axis=0)                     # suma sobre i de theta_{kj,i}^2
    return acum / acum.sum(axis=2, keepdims=True)          # [h, variable k, choque j]

def granger(Y, p, efecto, causas):
    """Prueba F de una ecuación y prueba de Wald robusta (HC0) de que los rezagos de 'causas' son cero."""
    y, X = rezagos(Y, p); y = y[:, efecto]; n, q = len(y), X.shape[1]; K = Y.shape[1]
    quitar = [1 + K * (j - 1) + c for j in range(1, p + 1) for c in causas]
    b = np.linalg.solve(X.T @ X, X.T @ y); e = y - X @ b; ssr_u = e @ e
    Xr = np.delete(X, quitar, axis=1); br = np.linalg.lstsq(Xr, y, rcond=None)[0]; er = y - Xr @ br
    r = len(quitar); F = ((er @ er - ssr_u) / r) / (ssr_u / (n - q))
    XtXi = np.linalg.inv(X.T @ X); V = XtXi @ (X.T * e**2) @ X @ XtXi           # HC0
    bR = b[quitar]; W = bR @ np.linalg.solve(V[np.ix_(quitar, quitar)], bR)
    return {"F": F, "gl": (r, n - q), "pF": stats.f.sf(F, r, n - q), "W": W, "pW": stats.chi2.sf(W, r)}

def bootstrap_irf(Y, p, h, B=500, semilla=1, orden=None):
    """Bandas al 90 % por bootstrap de residuos (percentiles) para las respuestas ortogonalizadas."""
    Y = np.asarray(Y, float); orden = list(range(Y.shape[1])) if orden is None else orden; Y = Y[:, orden]
    m = var_mco(Y, p); rng = np.random.default_rng(semilla); n, K = m["n"], m["K"]; sal = []
    for _ in range(B):
        U = m["U"][rng.integers(0, n, n)]; Yb = np.empty_like(Y); Yb[:p] = Y[:p]
        for t in range(p, len(Y)):
            Yb[t] = m["B"][0] + sum(m["A"][j - 1] @ Yb[t - j] for j in range(1, p + 1)) + U[t - p]
        mb = var_mco(Yb, p); sal.append(irf(mb["A"], h, np.linalg.cholesky(mb["S"])))
    sal = np.array(sal)
    return np.percentile(sal, 5, axis=0), np.percentile(sal, 95, axis=0)
