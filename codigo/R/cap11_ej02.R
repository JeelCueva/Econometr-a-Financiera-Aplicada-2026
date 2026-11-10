# Capítulo 11, Ejercicio 11.2: prueba de Johansen y VECM para el Brent y el WTI
source("codigo/R/cap11_funciones.R"); source("codigo/R/cap10_funciones.R")
m <- crudo_mensual(); L <- log(as.matrix(m[, c("Price_b", "Price_w")])); colnames(L) <- c("Brent", "WTI")
ic <- seleccion(L, 8); p <- unname(ic[which.min(ic[, 3]), 1]); k <- p - 1
cat(sprintf("VAR en niveles: BIC elige p = %d (AIC elige %d), de modo que k = p - 1 = %d rezagos de las diferencias\n\n",
            p, ic[which.min(ic[, 2]), 1], k))
J <- johansen(L, k); K <- 2
cat(sprintf("autovalores: %.5f  %.5f  (n = %d)\n", J$lam[1], J$lam[2], J$n))
cat("H0          traza   VC 95 %    máx. autovalor   VC 95 %\n")
for (r in 0:1) cat(sprintf("r <= %d   %8.3f  %8.3f        %8.3f      %8.3f\n", r, J$traza[r + 1], vc_traza[K - r, 2],
                           J$maxv[r + 1], vc_maxv[K - r, 2]))
v <- vecm(L, k, 1)
cat(sprintf("\nvector de cointegración normalizado: beta = (1, %.4f)\n", v$beta[2]))
cat("coeficientes de ajuste:\n")
for (i in 1:2) cat(sprintf("  alpha_%-5s = %8.4f (ee %.4f, t = %5.2f)\n", colnames(L)[i], v$alpha[i], v$ee_alpha[i], v$alpha[i] / v$ee_alpha[i]))
rho <- 1 + sum(v$beta * v$alpha)
cat(sprintf("persistencia del desequilibrio 1 + beta'alpha = %.4f; vida media = %.1f meses\n", rho, log(0.5) / log(rho)))
lr <- lr_beta(L, k, matrix(c(1, -1)))
cat(sprintf("\nH0: beta = (1, -1): LR = %.3f, p = %.2e (chi-cuadrado con 1 g.l.)\n", lr["LR"], pchisq(lr["LR"], 1, lower.tail = FALSE)))
