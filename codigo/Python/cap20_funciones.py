# Capítulo 20: funciones propias para aprendizaje automático en finanzas
import numpy as np, pandas as pd

VARS = ["dp", "ep", "de", "cape", "lty", "tbl", "tms", "infl", "mom12", "vol12"]

def cargar():
    """Predictores del mes t (filas) y exceso de rendimiento del mercado del mes t + 1 (%)."""
    d = pd.read_csv("datos/internacional/predictores_mensuales_1927_2018.csv")
    return d["mes"].to_numpy()[1:], d[VARS].to_numpy()[:-1], d["exceso"].to_numpy()[1:]

def estandarizar(Xtr, Xte):
    m, s = Xtr.mean(0), Xtr.std(0); return (Xtr - m) / s, (Xte - m) / s

def red_elastica(X, y, lam, alfa, b0=None, tol=1e-9, maxit=10000):
    """Descenso por coordenadas para (1/2n)||y - Xb||^2 + lam [alfa ||b||_1 + (1 - alfa)/2 ||b||^2].
    X estandarizada (varianza 1 dividiendo entre n) e y centrada. alfa = 1: lasso; alfa = 0: ridge."""
    n, p = X.shape; b = np.zeros(p) if b0 is None else b0.copy(); r = y - X @ b
    for it in range(maxit):
        cambio = 0.0
        for j in range(p):
            z = X[:, j] @ r / n + b[j]
            nb = np.sign(z) * max(abs(z) - lam * alfa, 0.0) / (1 + lam * (1 - alfa))
            if nb != b[j]: r -= X[:, j] * (nb - b[j]); cambio = max(cambio, abs(nb - b[j])); b[j] = nb
        if cambio < tol: break
    return b

def malla_lambda(X, y, alfa, k=30):
    lmax = np.abs(X.T @ y).max() / (len(y) * max(alfa, 0.01)); return lmax * np.logspace(0, -4, k)

def ajustar_validado(X, y, alfa, nval=120):
    """Elige lambda con los últimos nval meses de la muestra de entrenamiento como validación (sin mezclar el futuro)."""
    Xa, ya = X[:-nval], y[:-nval]; Xv, yv = X[-nval:], y[-nval:]
    Xas, Xvs = estandarizar(Xa, Xv); ma = ya.mean(); malla = malla_lambda(Xas, ya - ma, alfa)
    b = None; ecm = []
    for l in malla:
        b = red_elastica(Xas, ya - ma, l, alfa, b); ecm.append(((yv - ma - Xvs @ b)**2).mean())
    lam = malla[int(np.argmin(ecm))]
    Xs, _ = estandarizar(X, X); b = red_elastica(Xs, y - y.mean(), lam, alfa)
    return lam, b

def r2_os(y, f, fb):
    return 1 - ((y - f)**2).sum() / ((y - fb)**2).sum()

def clark_west(y, f, fb):
    """Estadístico de Clark y West (2007) contra el modelo de media histórica; valor p unilateral."""
    from scipy.stats import norm
    a = (y - fb)**2 - ((y - f)**2 - (fb - f)**2); t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a))); return t, norm.sf(t)

# de = dp - ep y tms = lty - tbl son combinaciones lineales exactas: los modelos multivariados usan 8 predictores
VARS_MULT = ["dp", "ep", "cape", "lty", "tbl", "infl", "mom12", "vol12"]
IDX_MULT = [VARS.index(v) for v in VARS_MULT]

def logit(y, X, tol=1e-12, maxit=100):
    """Logit por Newton-Raphson. Devuelve coeficientes, matriz de covarianza (inversa del hessiano) y log-verosimilitud."""
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        p = 1 / (1 + np.exp(-X @ b)); g = X.T @ (y - p); H = (X.T * (p * (1 - p))) @ X; d = np.linalg.solve(H, g); b += d
        if np.abs(d).max() < tol: break
    p = 1 / (1 + np.exp(-X @ b)); H = (X.T * (p * (1 - p))) @ X
    return b, np.linalg.inv(H), np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def auc(y, s):
    """Área bajo la curva ROC = Pr(s de un positivo > s de un negativo) (Mann-Whitney)."""
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def mejor_corte(X, y, minhoja):
    """Mejor partición binaria (variable, umbral) por suma de cuadrados de los residuos; None si no hay ninguna válida."""
    n, p = X.shape; mejor = (np.inf, None, None)
    for j in range(p):
        o = np.argsort(X[:, j], kind="stable"); xs, ys = X[o, j], y[o]; c1, c2 = np.cumsum(ys), np.cumsum(ys**2)
        for k in range(minhoja, n - minhoja + 1):
            if xs[k - 1] == xs[k]: continue
            sl, sr = c1[k - 1], c1[-1] - c1[k - 1]; sse = (c2[k - 1] - sl**2 / k) + (c2[-1] - c2[k - 1] - sr**2 / (n - k))
            if sse < mejor[0]: mejor = (sse, j, (xs[k - 1] + xs[k]) / 2)
    return mejor

def arbol(X, y, prof, minhoja=24):
    """Árbol de regresión (CART) de profundidad máxima prof, como diccionario anidado."""
    if prof == 0 or len(y) < 2 * minhoja: return {"hoja": y.mean(), "n": len(y)}
    sse, j, u = mejor_corte(X, y, minhoja)
    if j is None: return {"hoja": y.mean(), "n": len(y)}
    iz = X[:, j] <= u
    return {"var": j, "umbral": u, "n": len(y), "izq": arbol(X[iz], y[iz], prof - 1, minhoja), "der": arbol(X[~iz], y[~iz], prof - 1, minhoja)}

def predecir(t, X):
    if "hoja" in t: return np.full(len(X), t["hoja"])
    iz = X[:, t["var"]] <= t["umbral"]; out = np.empty(len(X))
    out[iz] = predecir(t["izq"], X[iz]); out[~iz] = predecir(t["der"], X[~iz]); return out

def imprimir_arbol(t, nombres, sangria="  "):
    if "hoja" in t: print(f"{sangria}hoja: media = {t['hoja']:.4f} (n = {t['n']})"); return
    print(f"{sangria}si {nombres[t['var']]} <= {t['umbral']:.4f}:"); imprimir_arbol(t["izq"], nombres, sangria + "    ")
    print(f"{sangria}si {nombres[t['var']]} > {t['umbral']:.4f}:"); imprimir_arbol(t["der"], nombres, sangria + "    ")
