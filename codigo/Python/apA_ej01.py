# Apéndice A, Ejercicio A.1: error estándar de la razón media/desviación por el método delta
import numpy as np, pandas as pd

lee = lambda f, c: pd.read_csv(f"datos/peru/{f}", parse_dates=["fecha"], index_col="fecha")[c]
series = {"EPU": lee("epu_yahoo.csv", "cierre_ajustado"), "SP500": lee("gspc_yahoo.csv", "cierre")}
for nom, p in series.items():
    p = p[:"2025-12-31"]
    pm = p.groupby(p.index.to_period("M")).last()
    R = pm.pct_change().dropna().to_numpy()                # rendimientos simples mensuales 2015-2025
    T = len(R); mu = R.mean(); d = R - mu
    s2 = np.mean(d**2); s = np.sqrt(s2)
    g3 = np.mean(d**3) / s**3; g4 = np.mean(d**4) / s2**2  # asimetría y curtosis
    th = mu / s
    v_norm = 1 + th**2 / 2                                 # varianza asintótica bajo normalidad
    v_gen = 1 - th * g3 + th**2 * (g4 - 1) / 4             # varianza asintótica general
    print(f"{nom}: T = {T}, media = {mu:.5f}, desv = {s:.5f}, asimetría = {g3:.3f}, curtosis = {g4:.3f}")
    print(f"   theta mensual = {th:.4f};  anualizada x raiz(12) = {th*np.sqrt(12):.4f}")
    print(f"   EE (normal)  = {np.sqrt(v_norm/T):.4f}   IC 95%: [{th-1.96*np.sqrt(v_norm/T):.4f}, {th+1.96*np.sqrt(v_norm/T):.4f}]")
    print(f"   EE (general) = {np.sqrt(v_gen/T):.4f}   IC 95%: [{th-1.96*np.sqrt(v_gen/T):.4f}, {th+1.96*np.sqrt(v_gen/T):.4f}]")
