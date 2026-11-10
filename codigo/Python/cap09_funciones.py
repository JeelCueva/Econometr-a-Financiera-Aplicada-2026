# Capítulo 9: funciones propias para GARCH, GJR-GARCH, EGARCH y GARCH en media (máxima verosimilitud)
# Vector de parámetros: mu, [lambda si media = "m"], omega, alpha, [gamma si GJR o EGARCH], beta, [nu si dist = "t"]
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

RAIZ_2_PI = np.sqrt(2 / np.pi)                          # E|z| para z normal estándar

def backcast(e):                                        # igual que en el Capítulo 8
    tau = min(75, len(e)); w = 0.94**np.arange(tau); w /= w.sum()
    return (w * e[:tau]**2).sum()

def desarmar(p, modelo, media):
    k = 2 if media == "m" else 1
    mu, lam = p[0], (p[1] if media == "m" else 0.0)
    if modelo == "garch": om, a, g, b = p[k], p[k + 1], 0.0, p[k + 2]; k += 3
    else:                 om, a, g, b = p[k:k + 4]; k += 4
    nu = p[k] if len(p) > k else None
    return mu, lam, om, a, g, b, nu

def filtro(p, r, bc, modelo="garch", media="c"):
    """Devuelve las innovaciones e_t y las varianzas condicionales sigma2_t."""
    mu, lam, om, a, g, b, _ = desarmar(p, modelo, media)
    n = len(r); e = np.empty(n); s2 = np.empty(n)
    for t in range(n):
        if modelo == "egarch":
            if t == 0: ls = om + b * np.log(bc)
            else:
                z = e[t - 1] / np.sqrt(s2[t - 1])
                ls = om + a * (abs(z) - RAIZ_2_PI) + g * z + b * ls
            s2[t] = np.exp(ls)
        else:                                           # GARCH (g = 0) y GJR-GARCH
            if t == 0: s2[t] = om + (a + g / 2 + b) * bc
            else:      s2[t] = om + (a + g * (e[t - 1] < 0)) * e[t - 1]**2 + b * s2[t - 1]
        e[t] = r[t] - mu - lam * s2[t]
    return e, s2

def llt(p, r, bc, modelo="garch", media="c", dist="normal"):
    e, s2 = filtro(p, r, bc, modelo, media)
    if dist == "normal":
        return -0.5 * (np.log(2 * np.pi) + np.log(s2) + e**2 / s2)
    nu = p[-1]
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2)) - 0.5 * np.log(s2)
            - (nu + 1) / 2 * np.log(1 + e**2 / (s2 * (nu - 2))))

def ajustar(r, modelo="garch", media="c", dist="normal"):
    bc = backcast(r - r.mean()); v = r.var(ddof=1)
    p0 = [r.mean()]; lim = [(None, None)]
    if media == "m": p0 += [0.0]; lim += [(-0.5, 0.5)]
    if modelo == "garch":  p0 += [0.05 * v, 0.08, 0.90];            lim += [(1e-8, None), (0, 1), (0, 1)]
    if modelo == "gjr":    p0 += [0.05 * v, 0.03, 0.08, 0.88];      lim += [(1e-8, None), (0, 1), (-1, 1), (0, 1)]
    if modelo == "egarch": p0 += [0.03 * np.log(v), 0.10, -0.08, 0.97]; lim += [(None, None), (0, 2), (-1, 1), (0, 0.9999)]
    if dist == "t": p0 += [8.0]; lim += [(2.05, 200)]
    def f(p):                                           # penaliza parámetros que hacen explotar la recursión
        with np.errstate(all="ignore"): v = -llt(p, r, bc, modelo, media, dist).sum()
        return v if np.isfinite(v) else 1e12
    o = minimize(f, p0, method="L-BFGS-B", bounds=lim, options={"ftol": 1e-15, "gtol": 1e-9, "maxiter": 3000})
    p = o.x; k = len(p)
    L = lambda q: llt(q, r, bc, modelo, media, dist).sum()
    def hess(p):
        h = 1e-5 * np.maximum(np.abs(p), 1); E = np.eye(k) * h
        g = np.array([(L(p + E[i]) - L(p - E[i])) / (2 * h[i]) for i in range(k)])
        H = np.array([[(L(p + E[i] + E[j]) - L(p + E[i] - E[j]) - L(p - E[i] + E[j]) + L(p - E[i] - E[j]))
                       / (4 * h[i] * h[j]) for j in range(k)] for i in range(k)])
        return g, H
    dentro = lambda q: all((lo is None or v >= lo) and (hi is None or v <= hi) for v, (lo, hi) in zip(q, lim))
    for _ in range(20):                                 # pulido: Newton con reducción del paso
        g, H = hess(p); paso = np.linalg.solve(H, g); mejora = False
        for _ in range(12):
            q = p - paso
            if dentro(q) and np.isfinite(L(q)) and L(q) > L(p): mejora = True; break
            paso = paso / 2
        if not mejora or L(q) - L(p) < 1e-10: p = q if mejora else p; break
        p = q
    h = 1e-5 * np.maximum(np.abs(p), 1); E = np.eye(k) * h
    G = np.column_stack([(llt(p + E[i], r, bc, modelo, media, dist) - llt(p - E[i], r, bc, modelo, media, dist))
                         / (2 * h[i]) for i in range(k)])
    H = np.array([[(L(p + E[i] + E[j]) - L(p + E[i] - E[j]) - L(p - E[i] + E[j]) + L(p - E[i] - E[j]))
                   / (4 * h[i] * h[j]) for j in range(k)] for i in range(k)])
    Hi = np.linalg.inv(-H)
    nm = ["mu"] + (["lambda"] if media == "m" else []) + ["omega", "alpha"] + \
         (["gamma"] if modelo != "garch" else []) + ["beta"] + (["nu"] if dist == "t" else [])
    e, s2 = filtro(p, r, bc, modelo, media)
    return {"p": p, "nombres": nm, "ll": L(p), "bc": bc, "e": e, "s2": s2, "k": k,
            "ee_rob": np.sqrt(np.diag(Hi @ G.T @ G @ Hi)), "ee": np.sqrt(np.diag(Hi))}
