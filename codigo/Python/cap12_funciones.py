# Capítulo 12: funciones propias para volatilidad multivariada (EWMA, CCC y DCC en dos etapas)
import numpy as np, pandas as pd
from scipy.optimize import minimize
from cap08_funciones import garch_fit

def panel_diario():
    """Rendimientos logarítmicos diarios (%) del S&P 500, el Nasdaq y el Brent en los días comunes."""
    s = pd.read_csv("datos/internacional/sp500_diario_yahoo.csv")[["Date", "Adj Close"]].rename(columns={"Adj Close": "sp"})
    n = pd.read_csv("datos/internacional/nasdaq_diario_yahoo.csv")[["Date", "Adj Close"]].rename(columns={"Adj Close": "nq"})
    b = pd.read_csv("datos/internacional/brent_diario.csv").rename(columns={"Price": "br"})
    d = s.merge(n, on="Date").merge(b, on="Date")
    R = 100 * np.diff(np.log(d[["sp", "nq", "br"]].to_numpy()), axis=0)
    return d["Date"].to_numpy()[1:], R

def ewma_cov(R, lam=0.94):
    """H_t = lam H_{t-1} + (1 - lam) e_{t-1} e_{t-1}', con H_1 = covarianza muestral de los primeros 30 días."""
    E = R - R.mean(0); T, K = E.shape; H = np.empty((T, K, K)); H[0] = np.cov(E[:30].T)
    for t in range(1, T): H[t] = lam * H[t - 1] + (1 - lam) * np.outer(E[t - 1], E[t - 1])
    return H

def corr_de_cov(H):
    d = 1 / np.sqrt(np.diagonal(H, axis1=-2, axis2=-1)); return H * d[..., :, None] * d[..., None, :]

def etapa1(R):
    """GARCH(1,1) normal para cada serie (Capítulo 8): residuos estandarizados y varianzas."""
    Z, S2, par = [], [], []
    for i in range(R.shape[1]):
        m = garch_fit(R[:, i], "normal"); Z.append((R[:, i] - m["p"][0]) / np.sqrt(m["s2"])); S2.append(m["s2"]); par.append(m)
    return np.column_stack(Z), np.column_stack(S2), par

def dcc_filtro(p, Z, Qb=None):
    """Q_t = (1-a-b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1};  R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)."""
    a, b = p; T, K = Z.shape; Qb = Z.T @ Z / T if Qb is None else Qb; Q = Qb.copy(); Rt = np.empty((T, K, K))
    for t in range(T):
        if t > 0: Q = (1 - a - b) * Qb + a * np.outer(Z[t - 1], Z[t - 1]) + b * Q
        Rt[t] = corr_de_cov(Q)
    return Rt

def ll_corr(Rt, Z):
    """Parte de correlación de la log-verosimilitud: -1/2 sum[ ln|R_t| + z_t' R_t^{-1} z_t - z_t' z_t ]."""
    sgn, ld = np.linalg.slogdet(Rt); q = np.einsum("ti,ti->t", Z, np.linalg.solve(Rt, Z[..., None])[..., 0])
    return -0.5 * (ld + q - (Z**2).sum(1))

def dcc_fit(Z):
    def f(p):
        if p[0] + p[1] >= 0.9999: return 1e10            # restricción a + b < 1
        return -ll_corr(dcc_filtro(p, Z), Z).sum()
    o = minimize(f, [0.03, 0.95], method="L-BFGS-B", bounds=[(1e-6, 0.5), (0.0, 0.9999)],
                 options={"ftol": 1e-14, "gtol": 1e-9})
    p = o.x; h = 1e-5 * np.maximum(np.abs(p), 1); E = np.eye(2) * h
    L = lambda q: ll_corr(dcc_filtro(q, Z), Z).sum()
    H = np.array([[(L(p + E[i] + E[j]) - L(p + E[i] - E[j]) - L(p - E[i] + E[j]) + L(p - E[i] - E[j]))
                   / (4 * h[i] * h[j]) for j in range(2)] for i in range(2)])
    Rc = np.corrcoef(Z.T); llc = ll_corr(np.broadcast_to(Rc, (len(Z),) + Rc.shape), Z).sum()
    return {"a": p[0], "b": p[1], "ee": np.sqrt(np.diag(np.linalg.inv(-H))), "ll": L(p), "ll_ccc": llc,
            "R": dcc_filtro(p, Z), "Rc": Rc}
