# Capítulo 18, Ejercicio 18.1: medidas de dependencia del S&P 500 y el Nasdaq (1999-2018)
source("codigo/R/cap18_funciones.R")
D <- indices(); f <- D$fecha; x <- D$x; y <- D$y; n <- length(x); u <- pseudo(x); v <- pseudo(y)
cat(sprintf("Rendimientos logarítmicos diarios (%%), %s a %s; n = %d\n", f[1], f[n], n))
for (k in 1:2) { z <- list(x, y)[[k]]; zc <- z - mean(z)
  cat(sprintf("  %s: media = %.4f; desv. est. = %.4f; asimetría = %.4f; curtosis = %.4f\n", c("S&P 500", "Nasdaq ")[k], mean(z), sd(z), mean(zc^3) / mean(zc^2)^1.5, mean(zc^4) / mean(zc^2)^2)) }
rp <- cor(x, y); tau <- cor(x, y, method = "kendall"); rs <- cor(x, y, method = "spearman")
cat(sprintf("\nPearson = %.4f; tau de Kendall = %.4f; rho de Spearman = %.4f\n", rp, tau, rs))
cat(sprintf("Spearman calculado con las pseudo-observaciones = Pearson de los rangos: %.4f\n", cor(u, v)))
rg <- sin(pi * tau / 2)
cat(sprintf("rho de la cópula gaussiana con el mismo tau: sin(pi tau / 2) = %.4f; Pearson de las normales de rangos = %.4f\n", rg, cor(qnorm(u), qnorm(v))))
cat("\nDependencia en las colas: Pr(V <= q | U <= q) y Pr(V > 1 - q | U > 1 - q)\n")
cat("     q     inferior   superior   gaussiana con rho = sin(pi tau / 2)\n")
for (q in c(0.10, 0.05, 0.01)) { l <- colas_empiricas(u, v, q); g <- phi2_diag(qnorm(q), rg) / q
  cat(sprintf("  %5.2f   %8.4f   %8.4f   %8.4f\n", q, l[1], l[2], g)) }
cat("\nPeores 5 días del S&P 500 y el rango del Nasdaq ese día:\n")
for (i in order(x)[1:5]) cat(sprintf("  %s: S&P 500 = %8.4f %%; Nasdaq = %8.4f %% (pseudo-observación %.4f)\n", f[i], x[i], y[i], v[i]))
