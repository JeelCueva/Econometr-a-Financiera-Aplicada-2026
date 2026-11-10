# Capítulo 14: funciones propias para modelos de factores (regresiones de series de tiempo, GRS, Fama-MacBeth)
import numpy as np, pandas as pd
from scipy import stats

VALOR = ["S1V1", "S1V3", "S1V5", "S3V1", "S3V3", "S3V5", "S5V1", "S5V3", "S5V5"]
MOMENTUM = ["S1M1", "S1M3", "S1M5", "S3M1", "S3M3", "S3M5", "S5M1", "S5M3", "S5M5"]
INDUSTRIAS = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]

def datos():
    """Excesos de rendimiento mensuales (%) de las carteras y factores en %, 1949-01 a 2017-03."""
    f = pd.read_csv("datos/internacional/french_mensual_1949_2017.csv")
    ex = {c: 100 * (f[c] - f["RF"]) for c in VALOR + MOMENTUM + INDUSTRIAS}
    fac = {c: 100 * f[c] for c in ["MktRF", "SMB", "HML", "Mom"]}
    return f["dates"].to_numpy(), pd.DataFrame(ex), pd.DataFrame(fac)

def regs_ts(R, F):
    """MCO de cada cartera sobre una constante y los factores: alfas, betas, residuos y Sigma insesgada."""
    R, F = np.asarray(R, float), np.asarray(F, float); T, L = F.shape
    X = np.column_stack([np.ones(T), F]); B = np.linalg.solve(X.T @ X, X.T @ R); E = R - X @ B
    S = E.T @ E / (T - L - 1); ee = np.sqrt(np.outer(np.diag(np.linalg.inv(X.T @ X)), np.diag(S)))
    r2 = 1 - (E**2).sum(0) / ((R - R.mean(0))**2).sum(0)
    return {"alfa": B[0], "beta": B[1:].T, "t_alfa": B[0] / ee[0], "E": E, "S": S, "r2": r2}

def grs(R, F):
    """Prueba de Gibbons, Ross y Shanken (1989) de que todos los alfas son cero."""
    R, F = np.asarray(R, float), np.asarray(F, float); T, N = R.shape; L = F.shape[1]
    m = regs_ts(R, F); mu = F.mean(0); Om = np.cov(F.T, bias=True).reshape(L, L)
    a2 = m["alfa"] @ np.linalg.solve(m["S"], m["alfa"]); sf2 = mu @ np.linalg.solve(Om, mu)
    Fst = (T - N - L) / N * T / (T - L - 1) * a2 / (1 + sf2)
    return {"F": Fst, "p": stats.f.sf(Fst, N, T - N - L), "gl": (N, T - N - L), "sr2_f": sf2,
            "alfa_abs": np.abs(m["alfa"]).mean(), "a2_mv": a2 * T / (T - L - 1)}

def sr2(X):
    """Cuadrado de la razón de Sharpe máxima (cartera tangente) con medias y covarianza muestrales (MV)."""
    X = np.asarray(X, float); mu = X.mean(0); V = np.cov(X.T, bias=True).reshape(len(mu), len(mu))
    return mu @ np.linalg.solve(V, mu)

def fama_macbeth(R, F):
    """Dos pasos: betas de series de tiempo; luego una regresión de corte transversal por mes."""
    R, F = np.asarray(R, float), np.asarray(F, float); T, N = R.shape; L = F.shape[1]
    m = regs_ts(R, F); X = np.column_stack([np.ones(N), m["beta"]])
    G = np.linalg.solve(X.T @ X, X.T @ R.T).T                     # T x (1 + L): gamma_0t y lambda_t
    lam = G.mean(0); ee_fm = G.std(0, ddof=1) / np.sqrt(T)
    Om = np.cov(F.T, bias=True).reshape(L, L); c = lam[1:] @ np.linalg.solve(Om, lam[1:])
    A = np.linalg.solve(X.T @ X, X.T); Os = np.zeros((L + 1, L + 1)); Os[1:, 1:] = Om
    V_sh = (A @ m["S"] @ A.T * (1 + c) + Os) / T                   # corrección de Shanken (1992)
    ajuste = X @ lam; media = R.mean(0)
    r2 = 1 - ((media - ajuste)**2).sum() / ((media - media.mean())**2).sum()
    return {"lam": lam, "ee_fm": ee_fm, "ee_sh": np.sqrt(np.diag(V_sh)), "r2_ct": r2, "beta": m["beta"], "c": c}
