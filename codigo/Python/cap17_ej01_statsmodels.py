# Capítulo 17, Ejercicio 17.1 (verificación): el mismo modelo con statsmodels (MarkovRegression, máxima verosimilitud directa)
import numpy as np, statsmodels.api as sm
from cap17_funciones import mercado
f, y = mercado(); np.random.seed(17)
mod = sm.tsa.MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True)
res = mod.fit(search_reps=20, disp=False); p = res.params
print(f"log-verosimilitud = {res.llf:.4f}")
for k, v in zip(mod.param_names, p): print(f"  {k:10s} {v:10.4f}")
print("desviaciones estándar por régimen:", np.round(np.sqrt(p[[n.startswith('sigma2') for n in mod.param_names]]), 4))
