# Capítulo 20, Ejercicio 20.1: predicción de la prima de mercado con un predictor a la vez (Goyal y Welch, 2008)
source("codigo/R/cap20_funciones.R")
D <- cargar(); mes <- D$mes; X <- D$X; y <- D$y; n <- length(y); i0 <- which(mes == "1957-01")
cat(sprintf("Objetivo: exceso de rendimiento del mercado del mes siguiente (%%), %s a %s; n = %d\n", mes[1], mes[n], n))
cat(sprintf("Fuera de muestra: %s a %s (%d meses), ventana creciente, reestimación mensual\n\n", mes[i0], mes[n], n - i0 + 1))
fb <- sapply(i0:n, function(i) mean(y[1:(i - 1)])); yo <- y[i0:n]
cat("predictor   pendiente   t(White)   R2 dentro (%)   R2 fuera (%)   R2 fuera CT (%)   CW      p(CW)\n")
for (j in seq_along(VARS)) { x <- X[, j]; Z <- cbind(1, x); A <- solve(crossprod(Z)); b <- A %*% crossprod(Z, y); e <- as.vector(y - Z %*% b)
  V <- n / (n - 2) * A %*% crossprod(Z * e) %*% A; r2 <- 1 - sum(e^2) / sum((y - mean(y))^2)
  f <- sapply(i0:n, function(i) { Zi <- cbind(1, x[1:(i - 1)]); bi <- solve(crossprod(Zi), crossprod(Zi, y[1:(i - 1)])); bi[1] + bi[2] * x[i] })
  cw <- clark_west(yo, f, fb)
  cat(sprintf("  %-7s  %10.4f  %8.2f  %12.2f  %13.2f  %15.2f  %6.2f  %7.4f\n", VARS[j], b[2], b[2] / sqrt(V[2, 2]), 100 * r2, 100 * r2_os(yo, f, fb), 100 * r2_os(yo, pmax(f, 0), fb), cw[1], cw[2])) }
