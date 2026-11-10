# Capítulo 7: funciones propias (razón de varianzas de Lo-MacKinlay y prueba de rachas)
import numpy as np

def vr_lm(r, q):
    r = np.asarray(r, float); nq = len(r); mu = r.mean()
    sa = ((r - mu)**2).sum() / (nq - 1)                   # varianza de 1 periodo (insesgada)
    rq = np.convolve(r, np.ones(q), "valid")              # sumas móviles de q periodos (traslapadas)
    m = q * (nq - q + 1) * (1 - q / nq)                   # corrección de sesgo de Lo-MacKinlay
    vr = ((rq - q * mu)**2).sum() / m / sa
    z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * nq))
    e2 = (r - mu)**2; den = e2.sum()**2
    theta = sum((2 * (q - j) / q)**2 * nq * (e2[j:] * e2[:-j]).sum() / den for j in range(1, q))
    return {"VR": vr, "z": z, "zrob": (vr - 1) / np.sqrt(theta / nq)}

def rachas(x):                                            # x: arreglo lógico (True = sube)
    x = np.asarray(x, bool); n1 = x.sum(); n2 = (~x).sum(); n = n1 + n2
    R = 1 + (x[1:] != x[:-1]).sum()
    ER = 2 * n1 * n2 / n + 1
    VR = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n**2 * (n - 1))
    return {"n1": n1, "n2": n2, "R": R, "ER": ER, "z": (R - ER) / np.sqrt(VR)}
