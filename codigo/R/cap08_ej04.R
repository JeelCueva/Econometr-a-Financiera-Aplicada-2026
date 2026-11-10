# Capítulo 8, Ejercicio 8.4: pronóstico de la volatilidad del Brent con el GARCH(1,1)-t
source("codigo/R/cap08_funciones.R")
b <- read.csv("datos/internacional/brent_diario.csv"); r <- 100 * diff(log(b$Price)); T <- length(r)
m <- garch_fit(r, "t"); p <- m$par
pers <- p["alpha"] + p["beta"]; s2bar <- p["omega"] / (1 - pers)
eT <- r[T] - p["mu"]; s2T <- m$s2[T]
s2T1 <- p["omega"] + p["alpha"] * eT^2 + p["beta"] * s2T          # varianza de mañana
cat(sprintf("último dato: %s; r_T = %.3f %%; sigma2_T = %.4f\n", b$Date[T + 1], r[T], s2T))
cat(sprintf("sigma2_T+1 = %.4f + %.4f x %.4f + %.4f x %.4f = %.4f\n",
            p["omega"], p["alpha"], eT^2, p["beta"], s2T, s2T1))
cat(sprintf("varianza incondicional = %.3f (volatilidad anual %.1f %%)\n", s2bar, sqrt(252 * s2bar)))
h <- 1:252
s2h <- s2bar + pers^(h - 1) * (s2T1 - s2bar)                       # E_T[sigma2_{T+h}]
for (k in c(1, 5, 21, 63, 126, 252))
  cat(sprintf("h = %3d: E[sigma2] = %.3f  (vol. diaria %.3f %%; anualizada %.1f %%)\n",
              k, s2h[k], sqrt(s2h[k]), sqrt(252 * s2h[k])))
for (H in c(21, 252)) {
  v <- sum(s2h[1:H])
  cat(sprintf("varianza acumulada a %3d días = %6.0f -> volatilidad del período %.2f %% (regla de la raíz: %.2f %%)\n",
              H, v, sqrt(v), sqrt(H * s2T1)))
}
