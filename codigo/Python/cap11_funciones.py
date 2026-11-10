# Capítulo 11: funciones propias para cointegración (Engle-Granger, Johansen y VECM)
import numpy as np
from cap05_funciones import adf

def crudo_mensual():
    """Promedios mensuales de los precios diarios del Brent y del WTI (días en que ambos cotizan)."""
    import pandas as pd
    b = pd.read_csv("datos/internacional/brent_diario.csv"); w = pd.read_csv("datos/internacional/wti_diario.csv")
    d = b.merge(w, on="Date", suffixes=("_b", "_w")); d["mes"] = d["Date"].str[:7]
    m = d.groupby("mes")[["Price_b", "Price_w"]].mean()
    return m[m.index <= "2026-10"]                           # se excluye el mes incompleto de noviembre

def vc_engle_granger(T):
    """Valores críticos de MacKinnon (2010) para la prueba de Engle-Granger con 2 variables y constante."""
    tabla = {"1%": (-3.89644, -10.9519, -22.527), "5%": (-3.33613, -6.1101, -6.823), "10%": (-3.04445, -4.2412, -2.720)}
    return {k: b0 + b1 / T + b2 / T**2 for k, (b0, b1, b2) in tabla.items()}

def engle_granger(y, x):
    """Paso 1: MCO de y sobre (1, x). Paso 2: ADF sin constante sobre los residuos, rezagos por AIC."""
    X = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ b
    r = adf(u, det="n")
    return {"a": b[0], "b": b[1], "u": u, "t": r["t"], "p": r["p"], "gamma": r["gamma"], "vc": vc_engle_granger(len(y))}

def _resid(Y, Z):
    return Y if Z is None else Y - Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]

def johansen(Y, k):
    """Johansen con constante sin restringir y k rezagos de las diferencias."""
    Y = np.asarray(Y, float); dY = np.diff(Y, axis=0); n = len(dY) - k
    Z = np.column_stack([np.ones(n)] + [dY[k - j:len(dY) - j] for j in range(1, k + 1)])
    R0 = _resid(dY[k:], Z); R1 = _resid(Y[k:-1], Z)          # Delta y_t y y_{t-1}, depurados de Z
    S00, S01, S11 = R0.T @ R0 / n, R0.T @ R1 / n, R1.T @ R1 / n
    M = np.linalg.solve(S11, S01.T @ np.linalg.solve(S00, S01))
    lam, V = np.linalg.eig(M); lam, V = lam.real, V.real; o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
    V = V / np.sqrt(np.diag(V.T @ S11 @ V))                   # normalización beta' S11 beta = I
    K = Y.shape[1]
    traza = np.array([-n * np.log(1 - lam[r:]).sum() for r in range(K)])
    maxv = np.array([-n * np.log(1 - lam[r]) for r in range(K)])
    return {"lam": lam, "V": V, "traza": traza, "maxv": maxv, "n": n, "S00": S00, "S01": S01, "S11": S11,
            "R0": R0, "R1": R1, "Z": Z, "dY": dY[k:], "Ylag": Y[k:-1]}

# Valores críticos al 90, 95 y 99 % (MacKinnon, Haug y Michelis, 1999), constante sin restringir;
# fila = número de relaciones bajo H0 contadas desde K - r = 1
VC_TRAZA = {1: (2.7055, 3.8415, 6.6349), 2: (13.4294, 15.4943, 19.9349), 3: (27.0669, 29.7961, 35.4628)}
VC_MAXV = {1: (2.7055, 3.8415, 6.6349), 2: (12.2971, 14.2639, 18.5200), 3: (18.8928, 21.1314, 25.8650)}

def vecm(Y, k, r=1):
    """VECM: beta de Johansen normalizada (primer elemento = 1); alpha, Gamma y constante por MCO."""
    J = johansen(Y, k); beta = J["V"][:, :r] / J["V"][0, :r]
    ect = J["Ylag"] @ beta; X = np.column_stack([ect, J["Z"]])
    B = np.linalg.lstsq(X, J["dY"], rcond=None)[0]; U = J["dY"] - X @ B; n, q = X.shape
    S = U.T @ U / (n - q); XtXi = np.linalg.inv(X.T @ X)
    ee = np.sqrt(np.outer(np.diag(XtXi), np.diag(S)))
    return {"beta": beta, "alpha": B[:r].T, "ee_alpha": ee[:r].T, "B": B, "ee": ee, "U": U, "S": S, "J": J, "ect": ect}

def lr_beta(Y, k, H):
    """Prueba de razón de verosimilitudes de beta = H phi (r = 1): chi-cuadrado con K - s grados de libertad."""
    J = johansen(Y, k); n = J["n"]
    S11h, S01h = H.T @ J["S11"] @ H, J["S01"] @ H
    Mh = np.linalg.solve(S11h, S01h.T @ np.linalg.solve(J["S00"], S01h)); lh = np.sort(np.linalg.eigvals(Mh).real)[::-1]
    LR = n * (np.log(1 - lh[0]) - np.log(1 - J["lam"][0]))
    return LR, J["lam"][0], lh[0]
