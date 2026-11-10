# Capítulo 21, Etapa 2: exposición al mercado (CAPM diario con errores de Newey-West)
source("codigo/R/cap21_funciones.R")
D <- cargar(); f <- D$fecha; R <- D$R; n <- length(f); m <- R[, 5]; X <- cbind(1, m)
cat(sprintf("R_it = a_i + b_i R_mt + e_it con el S&P 500 como mercado; n = %d; Newey-West con 5 rezagos\n", n))
cat("acción   alfa anual (%)   t(alfa)    beta    t(beta = 1)    R2     vol. idiosincrática anual (%)\n")
B <- c()
for (j in 1:4) { r <- nw_mco(R[, j], X, 5); e <- R[, j] - X %*% r$b; B <- c(B, r$b[2])
  cat(sprintf("%-6s  %12.2f  %9.2f  %8.4f  %10.2f   %7.4f   %14.2f\n", NOMBRES[j], 252 * r$b[1], r$b[1] / r$ee[1], r$b[2], (r$b[2] - 1) / r$ee[2], r$r2, sqrt(252) * sqrt(sum(e^2) / (n - 2)))) }
rp <- R[, 1:4] %*% rep(0.25, 4); r <- nw_mco(rp, X, 5)
cat(sprintf("Cartera 1/N: alfa anual = %.2f %% (t = %.2f); beta = %.4f; R2 = %.4f; promedio de las betas = %.4f\n", 252 * r$b[1], r$b[1] / r$ee[1], r$b[2], r$r2, mean(B)))
k <- f < "2016-01-01"; b1 <- nw_mco(rp[k], X[k, ], 5)$b; b2 <- nw_mco(rp[!k], X[!k, ], 5)$b
cat(sprintf("Estabilidad: beta de la cartera 2014-2015 = %.4f; 2016-2018 = %.4f\n", b1[2], b2[2]))
