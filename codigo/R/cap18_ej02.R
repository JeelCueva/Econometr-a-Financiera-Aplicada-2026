# Capítulo 18, Ejercicio 18.2: estimación de cuatro cópulas por máxima verosimilitud canónica
source("codigo/R/cap18_funciones.R")
D <- indices(); x <- D$x; y <- D$y; n <- length(x); u <- pseudo(x); v <- pseudo(y)
A <- ajustar(u, v); tau <- cor(x, y, method = "kendall")
cat(sprintf("n = %d; tau de Kendall empírico = %.4f\n\n", n, tau))
cat("cópula      parámetro(s)             log-verosim.     AIC        tau   lambda_inf  lambda_sup\n")
for (nom in c("gaussiana", "t", "clayton", "gumbel")) {
  if (nom == "t") { p <- A$t[1:2]; ll <- A$t[3]; k <- 2; ps <- sprintf("rho = %.4f, nu = %.3f", p[1], p[2]) }
  else { p <- A[[nom]][1]; ll <- A[[nom]][2]; k <- 1; ps <- sprintf("%s = %.4f", if (nom == "gaussiana") "rho" else "theta", p) }
  tl <- tau_lambda(nom, p)
  cat(sprintf("%-10s  %-22s  %12.4f  %10.2f  %.4f  %9.4f  %9.4f\n", nom, ps, ll, -2 * ll + 2 * k, tl[1], tl[2], tl[3]))
}
cat("\nEstimadores por inversión del tau de Kendall:\n")
cat(sprintf("  gaussiana/t: rho = sin(pi tau / 2) = %.4f; Clayton: theta = 2 tau / (1 - tau) = %.4f; Gumbel: theta = 1 / (1 - tau) = %.4f\n", sin(pi * tau / 2), 2 * tau / (1 - tau), 1 / (1 - tau)))
l <- colas_empiricas(u, v, 0.01); cat(sprintf("Colas empíricas al 1 %%: inferior = %.4f; superior = %.4f\n", l[1], l[2]))
cat(sprintf("\nLR de la t frente a la gaussiana (nu -> infinito, en la frontera) = %.2f\n", 2 * (A$t[3] - A$gaussiana[2])))
cat("Perfil de la log-verosimilitud de la t:\n")
for (nn in c(3, 5, 10, 20, 30)) { d <- dorada(function(rr) ll_t(rr, nn, u, v), -0.99, 0.99)
  cat(sprintf("  nu = %2d: rho = %.4f; log-verosimilitud = %.4f\n", nn, d[1], d[2])) }
