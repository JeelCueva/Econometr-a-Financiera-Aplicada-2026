# Capítulo 13, Ejercicio 13.5: VaR a 10 días del Brent: regla de la raíz, varianza GARCH y simulación filtrada
source("codigo/R/cap13_funciones.R"); source("codigo/R/cap08_funciones.R")
b <- read.csv("datos/internacional/brent_diario.csv"); r <- 100 * diff(log(b$Price)); f <- b$Date[-1]; T <- length(r)
m <- garch_fit(r, "t"); q <- m$par; z <- (r - q["mu"]) / sqrt(m$s2); H <- 10; p <- 0.99; N <- 100000
cat(sprintf("GARCH(1,1)-t: mu = %.4f, omega = %.4f, alpha = %.4f, beta = %.4f, nu = %.3f\n", q["mu"], q["omega"], q["alpha"], q["beta"], q["nu"]))
set.seed(2026)
for (fecha in c("2019-12-31", "2026-11-03")) {
  t <- which(f == fecha)
  s2 <- q["omega"] + q["alpha"] * (r[t] - q["mu"])^2 + q["beta"] * m$s2[t]            # varianza del día siguiente
  v1 <- var_es_t(q["mu"], sqrt(s2), q["nu"], p)
  phi <- q["alpha"] + q["beta"]; sb <- q["omega"] / (1 - phi)
  VH <- sum(sb + phi^(0:(H - 1)) * (s2 - sb))                                         # varianza acumulada a 10 días
  sim <- numeric(N)                                                                     # simulación histórica filtrada
  for (i in 1:N) {
    h <- s2; acc <- 0
    for (k in 1:H) { e <- sqrt(h) * z[sample.int(T, 1)]; acc <- acc + q["mu"] + e; h <- q["omega"] + q["alpha"] * e^2 + q["beta"] * h }
    sim[i] <- -acc
  }
  fh <- var_es_hist(sim, p)
  cat(sprintf("\n%s: volatilidad diaria de mañana = %.3f %% (anualizada %.1f %%)\n", fecha, sqrt(s2), sqrt(252 * s2)))
  cat(sprintf("  VaR 99 %% a 1 día (t)                  = %6.2f %%;  ES = %6.2f %%\n", v1[1], v1[2]))
  cat(sprintf("  regla de la raíz: raíz(10) x VaR      = %6.2f %%\n", sqrt(H) * v1[1]))
  cat(sprintf("  varianza GARCH acumulada (normal)     = %6.2f %%   (raíz de la varianza a 10 días = %.3f %%)\n",
              -H * q["mu"] + qnorm(p) * sqrt(VH), sqrt(VH)))
  cat(sprintf("  simulación filtrada (%d trayectorias) = %6.2f %%;  ES = %6.2f %%\n", N, fh[1], fh[2]))
}
