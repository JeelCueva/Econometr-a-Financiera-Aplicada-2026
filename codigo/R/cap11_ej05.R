# Capítulo 11, Ejercicio 11.5: ¿ayuda la corrección de errores a pronosticar? VECM frente a VAR en diferencias
source("codigo/R/cap11_funciones.R")
m <- crudo_mensual(); L <- log(as.matrix(m[, c("Price_b", "Price_w")])); T <- nrow(L); k <- 2
t0 <- which(m$mes == "2005-01")                             # primer mes pronosticado
pv <- pd <- matrix(NA, T, 2)
for (t in t0:T) {                                            # ventana creciente: datos hasta t-1
  Y <- L[1:(t - 1), ]; dY <- diff(L)
  x_d <- c(1, dY[t - 2, ], dY[t - 3, ])                     # (1, Delta y_{t-1}, Delta y_{t-2})
  v <- vecm(Y, k); pv[t, ] <- c(L[t - 1, ] %*% v$beta, x_d) %*% v$B
  J <- v$J; Bd <- solve(crossprod(J$Z), crossprod(J$Z, J$dY)); pd[t, ] <- x_d %*% Bd
}
real <- diff(L)[(t0 - 1):(T - 1), ]; ev <- real - pv[t0:T, ]; ed <- real - pd[t0:T, ]; er <- real
cat(sprintf("Pronósticos a un mes de Delta log precio: %s a %s (%d meses)\n\n", m$mes[t0], m$mes[T], nrow(real)))
cw <- function(e1, e2, p1, p2) {                             # Clark-West: modelo 1 anidado en el 2
  f <- e1^2 - (e2^2 - (p1 - p2)^2); n <- length(f); u <- f - mean(f); L <- floor(4 * (n / 100)^(2 / 9)); s <- sum(u^2) / n
  for (j in 1:L) s <- s + 2 * (1 - j / (L + 1)) * sum(u[(j + 1):n] * u[1:(n - j)]) / n
  tt <- mean(f) / sqrt(s / n); c(t = tt, p = pnorm(tt, lower.tail = FALSE))
}
for (i in 1:2) {
  nm <- c("Brent", "WTI")[i]
  cat(sprintf("%s: RECM (%%): paseo aleatorio = %.3f; VAR en diferencias = %.3f; VECM = %.3f\n", nm,
              100 * sqrt(mean(er[, i]^2)), 100 * sqrt(mean(ed[, i]^2)), 100 * sqrt(mean(ev[, i]^2))))
  a <- cw(ed[, i], ev[, i], pd[t0:T, i], pv[t0:T, i])
  cat(sprintf("  Clark-West, VAR en diferencias frente a VECM: t = %.2f (p = %.4f)\n", a["t"], a["p"]))
  cat(sprintf("  R2 fuera de muestra del VECM frente al VAR en diferencias: %.2f %%\n", 100 * (1 - sum(ev[, i]^2) / sum(ed[, i]^2))))
}
