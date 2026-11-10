# Capítulo 15, Ejercicio 15.1: hechos estilizados de la curva de rendimientos y sus componentes principales
source("codigo/R/cap15_funciones.R")
D <- cargar(); Y <- D$Y; T <- nrow(Y)
cat(sprintf("Rendimientos cupón cero de EE. UU. (McCulloch-Kwon), %s a %s, T = %d meses\n\n", D$fecha[1], D$fecha[T], T))
cat("Vencimiento   media   desv.est.   autocorr(1)   autocorr(12)   corr. con 1 mes\n")
for (j in seq_along(TAU)) {
  y <- Y[, j]; a <- acf(y, 12, plot = FALSE)$acf
  cat(sprintf("%4d meses   %6.3f   %7.3f   %10.4f   %11.4f   %12.4f\n", TAU[j], mean(y), sd(y), a[2], a[13], cor(y, Y[, 1])))
}
pend <- Y[, 10] - Y[, 3]
cat(sprintf("\npendiente r120 - r3: media = %.3f; meses con curva invertida (r120 < r3): %d (%.1f %%)\n", mean(pend), sum(pend < 0), 100 * mean(pend < 0)))
cat(sprintf("mínimo de la pendiente: %.3f (%s); máximo: %.3f (%s)\n", min(pend), D$fecha[which.min(pend)], max(pend), D$fecha[which.max(pend)]))
e <- eigen(cov(Y), symmetric = TRUE); V <- e$vectors[, 1:3]; V <- sweep(V, 2, sign(V[10, ]), "*")
cat("\nComponentes principales (covarianza): % de la varianza total:", sprintf("%.2f", 100 * e$values[1:3] / sum(e$values)), "\n")
cat("Cargas (vencimiento: CP1  CP2  CP3)\n")
for (j in seq_along(TAU)) cat(sprintf("  %4d: %7.3f %7.3f %7.3f\n", TAU[j], V[j, 1], V[j, 2], V[j, 3]))
