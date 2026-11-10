"""Capítulo 5: funciones para pruebas de raíz unitaria (programadas desde las fórmulas del texto)."""
import numpy as np

def adf_reg(y, p, det="c", inicio=None):
    """dy_t = [const] [+ tend] + gamma*y_{t-1} + sum_j delta_j dy_{t-j} + e_t (índices desde 0)."""
    y = np.asarray(y, float); dy = np.diff(y)
    inicio = p if inicio is None else inicio
    k = np.arange(inicio, len(dy))                         # filas de la regresión
    cols = [y[k]] + [dy[k - j] for j in range(1, p + 1)]
    if det in ("c", "ct"): cols.append(np.ones(len(k)))
    if det == "ct": cols.append(k + 1.0)
    X = np.column_stack(cols); b, *_ = np.linalg.lstsq(X, dy[k], rcond=None)
    e = dy[k] - X @ b; n = len(e)
    V = e @ e / (n - X.shape[1]) * np.linalg.inv(X.T @ X)
    return {"gamma": b[0], "t": b[0] / np.sqrt(V[0, 0]), "ssr": e @ e, "n": n,
            "kcoef": X.shape[1], "coef": b, "ee": np.sqrt(np.diag(V))}

def adf(y, det="c", pmax=None):
    """Prueba ADF con número de rezagos elegido por AIC en una muestra común."""
    T = len(y); pmax = int(np.floor(12 * (T / 100)**0.25)) if pmax is None else pmax
    aic = []
    for p in range(pmax + 1):
        r = adf_reg(y, p, det, inicio=pmax)
        aic.append(r["n"] * np.log(r["ssr"] / r["n"]) + 2 * r["kcoef"])
    p = int(np.argmin(aic)); r = adf_reg(y, p, det); r["p"] = p
    return r

def kpss(y, det="c", l=None):
    """Prueba KPSS con núcleo de Bartlett y l rezagos."""
    y = np.asarray(y, float); T = len(y); l = int(np.floor(12 * (T / 100)**0.25)) if l is None else l
    tt = np.arange(1, T + 1)
    e = y - y.mean() if det == "c" else y - np.polyval(np.polyfit(tt, y, 1), tt)
    S = np.cumsum(e)
    lrv = e @ e / T + sum(2 * (1 - j / (l + 1)) * (e[j:] @ e[:-j]) / T for j in range(1, l + 1))
    return {"eta": S @ S / (T**2 * lrv), "l": l}

# Valores críticos asintóticos (MacKinnon, 2010; Kwiatkowski et al., 1992)
VC_ADF = {"c": {"1%": -3.43, "5%": -2.86, "10%": -2.57}, "ct": {"1%": -3.96, "5%": -3.41, "10%": -3.13}}
VC_KPSS = {"c": {"10%": 0.347, "5%": 0.463, "1%": 0.739}, "ct": {"10%": 0.119, "5%": 0.146, "1%": 0.216}}
