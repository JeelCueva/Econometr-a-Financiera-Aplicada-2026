# Capítulo 18, Ejercicio 18.3: la cópula de los residuos estandarizados (EWMA) frente a la de los rendimientos
source("codigo/R/cap18_funciones.R")
D <- indices(); x <- D$x; y <- D$y; zx <- ewma_estandarizar(x); zy <- ewma_estandarizar(y)
k <- function(z) mean((z - mean(z))^4) / mean((z - mean(z))^2)^2
cat(sprintf("Residuos estandarizados con EWMA (lambda = 0,94): curtosis S&P 500 = %.4f; Nasdaq = %.4f\n", k(zx), k(zy)))
cat(sprintf("  desv. est. = %.4f y %.4f; Pearson = %.4f; tau de Kendall = %.4f\n", sd(zx), sd(zy), cor(zx, zy), cor(zx, zy, method = "kendall")))
R <- list("rendimientos" = list(pseudo(x), pseudo(y)), "residuos EWMA" = list(pseudo(zx), pseudo(zy)))
cat("\n                 lambda_inf(q)          lambda_sup(q)\n")
cat("                 10 %    5 %    1 %    10 %    5 %    1 %\n")
for (nom in names(R)) { u <- R[[nom]][[1]]; v <- R[[nom]][[2]]; L <- sapply(c(0.10, 0.05, 0.01), function(q) colas_empiricas(u, v, q))
  cat(sprintf("  %-13s  %s  %s\n", nom, paste(sprintf("%.3f", L[1, ]), collapse = " "), paste(sprintf("%.3f", L[2, ]), collapse = " "))) }
cat("\ndatos           cópula      parámetro(s)               log-verosim.      AIC    lambda_inf lambda_sup\n")
for (nom in names(R)) { u <- R[[nom]][[1]]; v <- R[[nom]][[2]]; A <- ajustar(u, v)
  for (cc in c("gaussiana", "t", "clayton", "gumbel")) {
    if (cc == "t") { p <- A$t[1:2]; ll <- A$t[3]; kk <- 2; ps <- sprintf("rho = %.4f, nu = %.3f", p[1], p[2]) }
    else { p <- A[[cc]][1]; ll <- A[[cc]][2]; kk <- 1; ps <- sprintf("%s = %.4f", if (cc == "gaussiana") "rho" else "theta", p) }
    tl <- tau_lambda(cc, p)
    cat(sprintf("%-14s  %-10s  %-24s  %12.4f  %10.2f  %8.4f  %8.4f\n", nom, cc, ps, ll, -2 * ll + 2 * kk, tl[2], tl[3]))
  } }
