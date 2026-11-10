# Capítulo 10, Ejercicio 10.1: VAR para el S&P 500 y el VIX: selección del orden, estimación y estabilidad
source("codigo/R/cap10_funciones.R")
s <- read.csv("datos/internacional/sp500_diario_yahoo.csv"); v <- read.csv("datos/internacional/vix_diario.csv")
d <- merge(s[, c("Date", "Adj.Close")], v[, c("DATE", "CLOSE")], by.x = "Date", by.y = "DATE")
Y <- cbind(r = 100 * diff(log(d$Adj.Close)), dv = 100 * diff(log(d$CLOSE)))
cat(sprintf("T = %d días (%s a %s); medias: r = %.4f, dv = %.4f; desv. est.: %.4f y %.4f\n\n",
            nrow(Y), d$Date[2], d$Date[nrow(d)], mean(Y[, 1]), mean(Y[, 2]), sd(Y[, 1]), sd(Y[, 2])))
ic <- seleccion(Y, 10)
cat("  p      AIC      BIC       HQ\n"); for (i in 1:nrow(ic)) cat(sprintf("%3d %8.4f %8.4f %8.4f\n", ic[i, 1], ic[i, 2], ic[i, 3], ic[i, 4]))
cat(sprintf("orden elegido: AIC = %d, BIC = %d, HQ = %d\n\n", ic[which.min(ic[, 2]), 1], ic[which.min(ic[, 3]), 1], ic[which.min(ic[, 4]), 1]))
m <- var_mco(Y, 2); XtXi <- solve(crossprod(m$X))
nom <- c("constante", "r(t-1)", "dv(t-1)", "r(t-2)", "dv(t-2)")
for (k in 1:2) {
  ee <- sqrt(diag(XtXi) * m$S[k, k])
  cat(sprintf("Ecuación de %s:\n", colnames(Y)[k]))
  for (i in 1:5) cat(sprintf("  %-10s %9.4f  (ee %.4f, t = %6.2f)\n", nom[i], m$B[i, k], ee[i], m$B[i, k] / ee[i]))
}
cat("\nMatriz de covarianzas de los residuos:\n"); print(round(m$S, 4))
cat(sprintf("correlación de los residuos = %.4f\n", m$S[1, 2] / sqrt(m$S[1, 1] * m$S[2, 2])))
cat("módulos de los autovalores de la matriz compañera:", sprintf("%.4f", Mod(eigen(companera(m$A))$values)), "\n")
