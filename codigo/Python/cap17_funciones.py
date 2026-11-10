# Capítulo 17: funciones propias para modelos con cambios de régimen de Markov
import numpy as np, pandas as pd

def mercado():
    """Exceso de rendimiento mensual del mercado de EE. UU. (%), julio de 1926 a noviembre de 2018."""
    d = pd.read_csv("datos/internacional/french_mercado_mensual_1926_2018.csv")
    return d["dates"].to_numpy(), d["Mkt-RF"].to_numpy()

def normal(y, mu, s):
    return np.exp(-0.5 * ((y - mu) / s)**2) / (s * np.sqrt(2 * np.pi))

def filtro_hamilton(D, P, rho):
    """D: T x K densidades condicionales; P[i, j] = Pr(s_t = j | s_{t-1} = i); rho: probabilidades de s_1.
    Devuelve probabilidades filtradas, predichas y la log-verosimilitud."""
    T, K = D.shape; xf = np.zeros((T, K)); xp = np.zeros((T, K)); ll = 0.0
    for t in range(T):
        xp[t] = rho if t == 0 else xf[t - 1] @ P
        num = xp[t] * D[t]; f = num.sum(); xf[t] = num / f; ll += np.log(f)
    return xf, xp, ll

def suavizado_kim(xf, xp, P):
    """Probabilidades suavizadas Pr(s_t | y_1..y_T) y la suma de las probabilidades conjuntas de transición."""
    T, K = xf.shape; xs = np.zeros((T, K)); xs[-1] = xf[-1]; N = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        r = xs[t + 1] / xp[t + 1]
        xs[t] = xf[t] * (P @ r)
        N += (xf[t][:, None] * P) * r[None, :]
    return xs, N

def em_media_varianza(y, K, mu0, s0, P0, media_comun=False, tol=1e-10, maxit=10000):
    """EM para y_t = mu_{s_t} + sigma_{s_t} e_t. Con media_comun=True, solo cambia la varianza."""
    mu, s, P, rho = np.array(mu0, float), np.array(s0, float), np.array(P0, float), np.full(K, 1 / K)
    ll0 = -np.inf
    for it in range(maxit):
        D = np.column_stack([normal(y, mu[j], s[j]) for j in range(K)])
        xf, xp, ll = filtro_hamilton(D, P, rho); xs, N = suavizado_kim(xf, xp, P)
        if ll - ll0 < tol: break
        ll0 = ll; w = xs.sum(0)
        mu = np.full(K, (xs * y[:, None]).sum() / w.sum()) if media_comun else (xs * y[:, None]).sum(0) / w
        s = np.sqrt((xs * (y[:, None] - mu)**2).sum(0) / w)
        P = N / N.sum(1, keepdims=True); rho = xs[0]
    return dict(mu=mu, s=s, P=P, rho=rho, ll=ll, xf=xf, xp=xp, xs=xs, it=it)

def em_msiar(y, K, c0, phi0, s0, P0, tol=1e-10, maxit=10000):
    """EM para y_t = c_{s_t} + phi y_{t-1} + sigma e_t (intercepto cambiante; phi y sigma comunes),
    condicionado en la primera observación."""
    z, x = y[1:], y[:-1]; T = len(z)
    c, phi, s, P, rho = np.array(c0, float), float(phi0), float(s0), np.array(P0, float), np.full(K, 1 / K)
    ll0 = -np.inf
    for it in range(maxit):
        D = np.column_stack([normal(z, c[j] + phi * x, s) for j in range(K)])
        xf, xp, ll = filtro_hamilton(D, P, rho); xs, N = suavizado_kim(xf, xp, P)
        if ll - ll0 < tol: break
        ll0 = ll; w = xs.sum(0)
        # mínimos cuadrados ponderados: ecuaciones normales en (c_1, ..., c_K, phi)
        A = np.zeros((K + 1, K + 1)); b = np.zeros(K + 1)
        A[:K, :K] = np.diag(w); A[:K, K] = A[K, :K] = (xs * x[:, None]).sum(0); A[K, K] = (x**2).sum()
        b[:K] = (xs * z[:, None]).sum(0); b[K] = (x * z).sum()
        th = np.linalg.solve(A, b); c, phi = th[:K], th[K]
        s = np.sqrt((xs * (z[:, None] - c - phi * x[:, None])**2).sum() / T)
        P = N / N.sum(1, keepdims=True); rho = xs[0]
    return dict(c=c, phi=phi, s=s, P=P, rho=rho, ll=ll, xf=xf, xp=xp, xs=xs, it=it)

def ergodicas(P):
    """Probabilidades estacionarias: autovector izquierdo de P con autovalor 1."""
    K = P.shape[0]; A = np.vstack([P.T - np.eye(K), np.ones(K)]); b = np.r_[np.zeros(K), 1]
    return np.linalg.lstsq(A, b, rcond=None)[0]

def ll_mv(th, y, rho):
    """Log-verosimilitud del modelo de dos regímenes en media y varianza; th = (mu1, mu2, s1, s2, p11, p22)."""
    P = np.array([[th[4], 1 - th[4]], [1 - th[5], th[5]]])
    D = np.column_stack([normal(y, th[0], th[2]), normal(y, th[1], th[3])])
    return filtro_hamilton(D, P, rho)[2]

def hessiano(f, th, h=1e-4):
    """Hessiano numérico por diferencias centrales."""
    k = len(th); H = np.zeros((k, k)); e = np.eye(k) * h
    for i in range(k):
        for j in range(k):
            H[i, j] = (f(th + e[i] + e[j]) - f(th + e[i] - e[j]) - f(th - e[i] + e[j]) + f(th - e[i] - e[j])) / (4 * h * h)
    return H

def recesion_mensual(fechas):
    """1 si el mes está en recesión según el NBER (del mes siguiente al pico hasta el valle, inclusive)."""
    r = pd.read_csv("datos/internacional/recesiones_nber.csv"); d = np.zeros(len(fechas), int)
    for p, v in zip(r["pico"], r["valle"]): d[(fechas > p) & (fechas <= v)] = 1
    return d

def produccion():
    """Variación porcentual trimestral de la producción industrial de EE. UU., 1970T1 a 2019T2."""
    d = pd.read_csv("datos/internacional/produccion_industrial_eeuu_trimestral.csv")
    return d["trimestre"].to_numpy(), d["produccion"].to_numpy()

def recesion_trimestral(trim):
    """1 si al menos dos de los tres meses del trimestre están en recesión según el NBER."""
    meses = [f"{q[:4]}-{3 * (int(q[-1]) - 1) + k:02d}" for q in trim for k in (1, 2, 3)]
    return (recesion_mensual(np.array(meses)).reshape(-1, 3).sum(1) >= 2).astype(int)

def cuantil_mezcla(a, w, mu, s, lo=-100.0, hi=100.0, it=200):
    """Cuantil a de una mezcla de normales por bisección."""
    from scipy.stats import norm
    for _ in range(it):
        q = 0.5 * (lo + hi)
        if (w * norm.cdf((q - mu) / s)).sum() < a: lo = q
        else: hi = q
    return 0.5 * (lo + hi)

def kupiec_christoffersen(h, a):
    """LR de cobertura incondicional (Kupiec) y de independencia (Christoffersen) para una serie de excepciones."""
    xl = lambda n, p: n * np.log(p) if n > 0 else 0.0
    n, x = len(h), h.sum(); p = x / n
    uc = -2 * (xl(n - x, 1 - a) + xl(x, a) - xl(n - x, 1 - p) - xl(x, p))
    a0, a1 = h[:-1], h[1:]
    n00, n01 = ((a0 == 0) & (a1 == 0)).sum(), ((a0 == 0) & (a1 == 1)).sum(); n10, n11 = ((a0 == 1) & (a1 == 0)).sum(), ((a0 == 1) & (a1 == 1)).sum()
    p01, p11, p1 = n01 / (n00 + n01), (n11 / (n10 + n11) if n10 + n11 else 0.0), (n01 + n11) / (n00 + n01 + n10 + n11)
    ind = -2 * (xl(n00 + n10, 1 - p1) + xl(n01 + n11, p1) - xl(n00, 1 - p01) - xl(n01, p01) - xl(n10, 1 - p11) - xl(n11, p11))
    return uc, ind, n11
