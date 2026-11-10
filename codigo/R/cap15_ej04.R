# Capítulo 15, Ejercicio 15.4: Nelson-Siegel dinámico (Diebold y Li)
source("codigo/R/cap15_funciones.R")
D <- cargar(); Y <- D$Y; f <- D$fecha
malla <- seq(0.010, 0.200, by = 0.0005)
L <- ns_lambda(Y, TAU, malla)
cat(sprintf("lambda común (malla 0,010 a 0,200, paso 0,0005): %.4f  -> curvatura máxima en tau = %.1f meses\n", L$lam, 1.793 / L$lam))
for (lam in c(L$lam, 0.0609)) {
  a <- ns_ajuste(Y, TAU, lam); B <- a$B
  cat(sprintf("\nlambda = %.4f: RECM global = %.4f puntos porcentuales\n", lam, a$recm))
  cat("  RECM por vencimiento:", paste(sprintf("%d:%.3f", TAU, sqrt(colMeans((Y - a$A)^2))), collapse = "  "), "\n")
  cat("  factor     media    desv.est.  rho(1)   rho(12)\n")
  for (j in 1:3) { b <- B[, j]; ac <- acf(b, lag.max = 12, plot = FALSE)$acf
    cat(sprintf("  beta%d  %9.4f %9.4f %8.4f %8.4f\n", j - 1, mean(b), sd(b), ac[2], ac[13])) }
}
a <- ns_ajuste(Y, TAU, 0.0609); B <- a$B
niv <- Y[, 10]; pen <- Y[, 10] - Y[, 3]; cur <- 2 * Y[, 8] - Y[, 3] - Y[, 10]
cat("\nCorrelaciones (lambda = 0,0609) con los factores empíricos:\n")
cat(sprintf("  corr(beta0, y120)                 = %.4f\n", cor(B[, 1], niv)))
cat(sprintf("  corr(-beta1, y120 - y3)           = %.4f\n", cor(-B[, 2], pen)))
cat(sprintf("  corr(beta2, 2 y36 - y3 - y120)    = %.4f\n", cor(B[, 3], cur)))
cat("\nFactores y ajuste en fechas elegidas (lambda = 0,0609):\n")
for (d in c("1950-01", "1981-09", "1991-02")) { i <- which(f == d)
  cat(sprintf("  %s: beta0 = %.3f  beta1 = %.3f  beta2 = %.3f  RECM del mes = %.3f\n", d, B[i, 1], B[i, 2], B[i, 3], sqrt(mean((Y[i, ] - a$A[i, ])^2)))) }
