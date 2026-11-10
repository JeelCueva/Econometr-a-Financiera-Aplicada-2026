# Capítulo 17, Ejercicio 17.1: dos regímenes en la media y la varianza del mercado de EE. UU. (1926-2018)
source("codigo/R/cap17_funciones.R")
D <- mercado(); f <- D$fecha; y <- D$y; T <- length(y); m <- mean(y); sd0 <- sqrt(mean((y - m)^2))
cat(sprintf("Exceso de rendimiento mensual del mercado, %s a %s, T = %d\n", f[1], f[T], T))
r <- em_media_varianza(y, 2, c(m, m), c(0.7, 1.5) * sd0, matrix(c(0.95, 0.05, 0.05, 0.95), 2, 2))
cat(sprintf("EM: %d iteraciones; log-verosimilitud = %.4f\n", r$it, r$ll))
th <- c(r$mu, r$s, r$P[1, 1], r$P[2, 2])
H <- hessiano(function(t) ll_mv(t, y, r$rho), th); ee <- sqrt(diag(solve(-H)))
cat("\nparámetro   estimación     ee\n")
nom <- c("mu1", "mu2", "sigma1", "sigma2", "p11", "p22")
for (i in 1:6) cat(sprintf("  %-7s %10.4f %9.3f\n", nom[i], th[i], ee[i]))
pi <- ergodicas(r$P); P <- r$P
cat(sprintf("\nprobabilidades estacionarias: %.4f %.4f\n", pi[1], pi[2]))
cat(sprintf("duraciones esperadas (meses): %.2f %.2f\n", 1 / (1 - P[1, 1]), 1 / (1 - P[2, 2])))
cat(sprintf("autocorrelación del régimen, lambda = p11 + p22 - 1 = %.4f\n", P[1, 1] + P[2, 2] - 1))
mm <- sum(pi * r$mu); vv <- sum(pi * (r$s^2 + r$mu^2)) - mm^2
k4 <- sum(pi * (3 * r$s^4 + 6 * r$s^2 * (r$mu - mm)^2 + (r$mu - mm)^4)) / vv^2
yc <- y - mean(y); kd <- mean(yc^4) / mean(yc^2)^2
cat(sprintf("momentos implícitos: media = %.4f; desv. est. = %.4f; curtosis = %.4f\n", mm, sqrt(vv), k4))
cat(sprintf("momentos muestrales: media = %.4f; desv. est. = %.4f; curtosis = %.4f\n", mean(y), sd0, kd))
lln <- -0.5 * T * (log(2 * base::pi * sd0^2) + 1)
cat(sprintf("\nlog-verosimilitud de la normal = %.4f; diferencia = %.4f\n", lln, r$ll - lln))
cat(sprintf("razón de Sharpe mensual por régimen: %.4f %.4f\n", r$mu[1] / r$s[1], r$mu[2] / r$s[2]))
cat(sprintf("meses con Pr(régimen 2 | toda la muestra) > 0,5: %d de %d\n", sum(r$xs[, 2] > 0.5), T))
