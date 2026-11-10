# Capítulo 8: funciones propias para estimar un GARCH(1,1) por máxima verosimilitud
# Parámetros: p = (mu, omega, alpha, beta) y, si dist = "t", nu como quinto elemento.
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

def backcast(e):                          # varianza inicial: promedio ponderado de los primeros e^2
    tau = min(75, len(e)); w = 0.94**np.arange(tau); w /= w.sum()
    return (w * e[:tau]**2).sum()

def garch_s2(p, r, bc):                   # recursión sigma2_t = omega + alpha e_{t-1}^2 + beta sigma2_{t-1}
    e = r - p[0]; s2 = np.empty_like(e)
    s2[0] = p[1] + (p[2] + p[3]) * bc
    for t in range(1, len(e)):
        s2[t] = p[1] + p[2] * e[t - 1]**2 + p[3] * s2[t - 1]
    return s2

def garch_llt(p, r, bc, dist="normal"):   # log-verosimilitud de cada observación
    e = r - p[0]; s2 = garch_s2(p, r, bc)
    if dist == "normal":
        return -0.5 * (np.log(2 * np.pi) + np.log(s2) + e**2 / s2)
    nu = p[4]                             # t de Student estandarizada (varianza 1)
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2)) - 0.5 * np.log(s2)
            - (nu + 1) / 2 * np.log(1 + e**2 / (s2 * (nu - 2))))

def garch_fit(r, dist="normal"):
    bc = backcast(r - r.mean())
    p0 = [r.mean(), 0.05 * r.var(ddof=1), 0.08, 0.90]; lim = [(None, None), (1e-8, None), (0, 1), (0, 1)]
    if dist == "t": p0 += [8.0]; lim += [(2.05, 200)]
    o = minimize(lambda p: -garch_llt(p, r, bc, dist).sum(), p0, method="L-BFGS-B", bounds=lim,
                 options={"ftol": 1e-15, "gtol": 1e-9, "maxiter": 2000})
    p = o.x; k = len(p); h = 1e-5 * np.maximum(np.abs(p), 1); E = np.eye(k) * h
    G = np.column_stack([(garch_llt(p + E[i], r, bc, dist) - garch_llt(p - E[i], r, bc, dist)) / (2 * h[i])
                         for i in range(k)])                         # scores de cada observación
    L = lambda q: garch_llt(q, r, bc, dist).sum()
    H = np.array([[(L(p + E[i] + E[j]) - L(p + E[i] - E[j]) - L(p - E[i] + E[j]) + L(p - E[i] - E[j]))
                   / (4 * h[i] * h[j]) for j in range(k)] for i in range(k)])   # hessiana numérica
    Hi = np.linalg.inv(-H)
    nm = ["mu", "omega", "alpha", "beta", "nu"][:k]
    return {"par": dict(zip(nm, p)), "p": p, "ll": -o.fun, "bc": bc, "s2": garch_s2(p, r, bc),
            "ee": dict(zip(nm, np.sqrt(np.diag(Hi)))), "ee_rob": dict(zip(nm, np.sqrt(np.diag(Hi @ G.T @ G @ Hi))))}
