# Capítulo 15, Ejercicio 15.2: la hipótesis de las expectativas (regresiones de Campbell y Shiller)
source("codigo/R/cap15_funciones.R")
D <- cargar(); Y <- D$Y; colnames(Y) <- paste0("r", TAU); T <- nrow(Y)
cat("(a) Tasas cortas futuras: (1/k) sum_{i=0}^{k-1} y_{m,t+m i} - y_{m,t} = a + b (y_{k,t} - y_{m,t})\n")
cat("    H0 (expectativas): b = 1; errores de Newey-West con k rezagos\n")
cat("   m     k     n       b      ee(b)   t(b=1)     R2\n")
casos <- list(c(1, 3), c(1, 6), c(1, 12), c(12, 36), c(12, 60), c(12, 120))
for (cs in casos) {
  m <- cs[1]; k <- cs[2]; ym <- Y[, paste0("r", m)]; yk <- Y[, paste0("r", k)]; h <- k / m; n <- T - (k - m)
  izq <- sapply(1:n, function(t) mean(ym[t + m * (0:(h - 1))])) - ym[1:n]
  r <- mco_nw(izq, yk[1:n] - ym[1:n], k)
  cat(sprintf("%4d %5d %5d %8.4f %8.4f %8.2f %8.3f\n", m, k, n, r$b[2], r$ee[2], (r$b[2] - 1) / r$ee[2], r$r2))
}
cat("\n(b) Tasa larga: y_{11,t+1} - y_{12,t} = a + b (y_{12,t} - y_{1,t}) / 11;  H0: b = 1\n")
y12 <- Y[, "r12"]; y11 <- Y[, "r11"]; y1 <- Y[, "r1"]
r <- mco_nw(y11[-1] - y12[-T], (y12[-T] - y1[-T]) / 11, 1)
cat(sprintf("    b = %.4f (ee %.4f); t(b = 1) = %.2f; t(b = 0) = %.2f; R2 = %.4f; n = %d\n", r$b[2], r$ee[2], (r$b[2] - 1) / r$ee[2], r$b[2] / r$ee[2], r$r2, r$n))
