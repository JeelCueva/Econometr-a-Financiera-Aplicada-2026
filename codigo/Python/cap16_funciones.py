# Capítulo 16: funciones propias para estudios de eventos
import numpy as np, pandas as pd

ACT = ["AAPL", "AMZN", "FB", "GOOG"]

def cargar():
    """Rendimientos simples diarios (%) de las cuatro acciones y del S&P 500, 2014-2018."""
    d = pd.read_csv("datos/internacional/gafa_sp500_diario_2014_2018.csv"); P = d[ACT + ["SP500"]].to_numpy()
    return d["fecha"].to_numpy()[1:], 100 * (P[1:] / P[:-1] - 1)

def rendimientos_normales(r, m, t0, est=(-250, -11), evt=(-5, 5), modelo="mercado", excluir=()):
    """Rendimientos anormales en la ventana del evento y su matriz de covarianzas (error de predicción).
    modelo: 'mercado' (r = a + b m), 'media' (r = a) o 'ajustado' (r - m, sin parámetros).
    excluir: posiciones absolutas que se eliminan de la ventana de estimación."""
    ie = np.array([t for t in range(t0 + est[0], t0 + est[1] + 1) if t not in set(excluir)])
    iv = np.arange(t0 + evt[0], t0 + evt[1] + 1)
    y = r if modelo != "ajustado" else r - m
    if modelo == "mercado": X, Xs = np.column_stack([np.ones(len(ie)), m[ie]]), np.column_stack([np.ones(len(iv)), m[iv]])
    else: X, Xs = np.ones((len(ie), 1)), np.ones((len(iv), 1))
    if modelo == "ajustado": b = np.zeros(0); X, Xs = X[:, :0], Xs[:, :0]
    else: b = np.linalg.solve(X.T @ X, X.T @ y[ie])
    e = y[ie] - X @ b; k = X.shape[1]; s2 = (e @ e) / (len(ie) - k)
    AR = y[iv] - Xs @ b
    V = s2 * (np.eye(len(iv)) + (Xs @ np.linalg.solve(X.T @ X, Xs.T) if k else 0))
    return dict(AR=AR, V=V, s2=s2, b=b, gl=len(ie) - k, tau=np.arange(evt[0], evt[1] + 1), e=e)

def car(res, t1, t2):
    """CAR(t1, t2) y su desviación estándar: gamma' AR y sqrt(gamma' V gamma)."""
    g = ((res["tau"] >= t1) & (res["tau"] <= t2)).astype(float)
    return g @ res["AR"], np.sqrt(g @ res["V"] @ g)

def pruebas(AR, V, gl):
    """Pruebas de un día con N eventos: J1 (varianzas estimadas), Patell, BMP y signo.
    AR y V: vectores con el rendimiento anormal del día y su varianza para cada evento."""
    N = len(AR); SAR = AR / np.sqrt(V)
    J1 = AR.mean() / np.sqrt(V.sum() / N**2)
    patell = SAR.sum() / np.sqrt(np.sum(gl / (gl - 2)))
    bmp = SAR.mean() / (SAR.std(ddof=1) / np.sqrt(N))
    signo = ((AR > 0).sum() - N / 2) / np.sqrt(N / 4)
    return dict(CAAR=AR.mean(), J1=J1, Patell=patell, BMP=bmp, signo=signo, pos=(AR > 0).sum(), SAR=SAR)

def corrado(A):
    """Prueba de rangos de Corrado (1989). A: matriz N x L de rendimientos anormales (estimación + evento);
    la última columna es el día 0."""
    N, L = A.shape; K = np.argsort(np.argsort(A, axis=1), axis=1) + 1.0
    D = (K - (L + 1) / 2).mean(0)
    return D[-1] / np.sqrt((D**2).mean())

class LCG:
    """Generador congruencial de Park y Miller (1988): x <- 16807 x mod (2^31 - 1). Da la misma
    secuencia en R y en Python, lo que permite reproducir la simulación con resultados idénticos."""
    M = 2147483647
    def __init__(self, semilla): self.x = semilla
    def u(self):
        self.x = (16807 * self.x) % self.M; return self.x / self.M
