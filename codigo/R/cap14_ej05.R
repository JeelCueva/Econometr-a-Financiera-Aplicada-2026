# Capítulo 14, Ejercicio 14.5: factores estadísticos: componentes principales de 30 carteras
source("codigo/R/cap14_funciones.R")
D <- datos(); R <- D$R; T <- nrow(R)
e <- eigen(mcov(R), symmetric = TRUE); lam <- e$values; V <- e$vectors
V <- sweep(V, 2, sign(colSums(V)), "*")                    # signo: cargas que sumen positivo
cat("Proporción de la varianza total explicada:\n")
for (k in 1:5) cat(sprintf("  CP%d: %6.2f %%  (acumulada %6.2f %%)\n", k, 100 * lam[k] / sum(lam), 100 * sum(lam[1:k]) / sum(lam)))
P <- sweep(R, 2, colMeans(R)) %*% V[, 1:3]
cat("\nCorrelación de los tres primeros componentes con los factores:\n          Mkt     SMB     HML     Mom\n")
for (k in 1:3) cat(sprintf("CP%d   %6.3f  %6.3f  %6.3f  %6.3f\n", k, cor(P[, k], D$F[, 1]), cor(P[, k], D$F[, 2]), cor(P[, k], D$F[, 3]), cor(P[, k], D$F[, 4])))
cat("\nR2 de cada componente explicado por los cuatro factores:", sprintf("%.3f", sapply(1:3, function(k) summary(lm(P[, k] ~ D$F))$r.squared)), "\n")
cat(sprintf("\nR2 medio de las 30 carteras: con 3 componentes = %.3f; con Mkt, SMB y HML = %.3f\n",
            mean(regs_ts(R, P)$r2), mean(regs_ts(R, D$F[, 1:3])$r2)))
