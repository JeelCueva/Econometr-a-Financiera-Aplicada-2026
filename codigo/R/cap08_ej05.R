# Capítulo 8, Ejercicio 8.5: evaluación fuera de muestra de pronósticos de varianza (GARCH, EWMA, ventana móvil)
source("codigo/R/cap08_funciones.R")
b <- read.csv("datos/internacional/brent_diario.csv"); r <- 100 * diff(log(b$Price)); f <- b$Date[-1]
dentro <- f <= "2015-12-31"; n0 <- sum(dentro); T <- length(r)
m <- garch_fit(r[dentro], "normal"); p <- m$par              # parámetros fijos, estimados hasta 2015
e <- r - p["mu"]
h_garch <- garch_s2(p, r, m$bc)                               # el filtro corre sobre toda la muestra
h_ewma <- numeric(T); h_ewma[1] <- m$bc                       # RiskMetrics: lambda = 0,94
for (t in 2:T) h_ewma[t] <- 0.94 * h_ewma[t - 1] + 0.06 * e[t - 1]^2
h_vent <- rep(NA, T)                                          # varianza de los 250 días previos
for (t in 251:T) h_vent[t] <- mean(e[(t - 250):(t - 1)]^2)
fuera <- (n0 + 1):T; y <- e[fuera]^2
cat(sprintf("Estimación: %s a %s (%d días); evaluación: %s a %s (%d días)\n",
            f[1], f[n0], n0, f[n0 + 1], f[T], length(fuera)))
cat(sprintf("GARCH estimado: omega = %.4f, alpha = %.4f, beta = %.4f\n\n", p["omega"], p["alpha"], p["beta"]))
nw_t <- function(d) {                                         # t de Diebold-Mariano con Newey-West
  n <- length(d); L <- floor(4 * (n / 100)^(2 / 9)); u <- d - mean(d); s <- sum(u^2) / n
  for (j in 1:L) s <- s + 2 * (1 - j / (L + 1)) * sum(u[(j + 1):n] * u[1:(n - j)]) / n
  mean(d) / sqrt(s / n)
}
modelos <- list(GARCH = h_garch[fuera], EWMA = h_ewma[fuera], "Ventana 250" = h_vent[fuera])
qlike <- lapply(modelos, function(h) log(h) + y / h)
cat("Modelo        ECM      QLIKE    MZ: a       b      R2\n")
for (nm in names(modelos)) {
  h <- modelos[[nm]]; mz <- lm(y ~ h)
  cat(sprintf("%-11s %8.2f %8.4f %8.3f %7.3f %7.4f\n", nm, mean((y - h)^2), mean(qlike[[nm]]),
              coef(mz)[1], coef(mz)[2], summary(mz)$r.squared))
}
cat("\nDiebold-Mariano sobre QLIKE (d = pérdida del rival - pérdida del GARCH; t > 0 favorece al GARCH):\n")
for (nm in c("EWMA", "Ventana 250"))
  cat(sprintf("  GARCH frente a %-11s: media de d = %.4f, t = %.2f\n", nm,
              mean(qlike[[nm]] - qlike$GARCH), nw_t(qlike[[nm]] - qlike$GARCH)))
