# Capítulo 11, Ejercicio 11.4: el modelo de valor presente: ¿están cointegrados precios, dividendos y utilidades?
source("codigo/R/cap11_funciones.R"); source("codigo/R/cap10_funciones.R")
s <- read.csv("datos/internacional/sp500_mensual_shiller.csv")
s <- s[s$Date <= "2023-06-01" & substr(s$Date, 6, 7) == "01", ]       # enero de cada año
P <- log(s$Real.Price); D <- log(s$Real.Dividend); E <- log(s$Real.Earnings); T <- length(P)
cat(sprintf("Datos anuales (enero), %s a %s, T = %d; valores reales\n\n", substr(s$Date[1], 1, 4), substr(s$Date[T], 1, 4), T))
cat("ADF con constante (rezagos por AIC); valor crítico al 5 %: -2.86\n")
for (et in c("log P", "log D", "log E", "log(D/P)", "log(E/P)")) {
  y <- switch(et, "log P" = P, "log D" = D, "log E" = E, "log(D/P)" = D - P, "log(E/P)" = E - P)
  a <- adf(y, "c"); cat(sprintf("  %-9s t = %6.2f (p = %d)\n", et, a$t, a$p))
}
for (par in c("D", "E")) {
  Y <- cbind(P, if (par == "D") D else E); ic <- seleccion(Y, 4); k <- unname(ic[which.min(ic[, 3]), 1]) - 1
  J <- johansen(Y, k); v <- vecm(Y, k); lr <- lr_beta(Y, k, matrix(c(1, -1)))
  cat(sprintf("\nPrecio y %s: k = %d; traza r=0: %.2f (VC 95 %% 15.49); r<=1: %.2f (VC 3.84)\n", par, k, J$traza[1], J$traza[2]))
  cat(sprintf("  beta = (1, %.4f); alpha_P = %.4f (t = %.2f); alpha_%s = %.4f (t = %.2f)\n", v$beta[2],
              v$alpha[1], v$alpha[1] / v$ee_alpha[1], par, v$alpha[2], v$alpha[2] / v$ee_alpha[2]))
  cat(sprintf("  LR beta = (1, -1): %.2f (p = %.4f)\n", lr["LR"], pchisq(lr["LR"], 1, lower.tail = FALSE)))
}
cat(sprintf("\nrazón dividendo/precio: %.2f %% en %s y %.2f %% en %s\n", 100 * exp(D[1] - P[1]), substr(s$Date[1], 1, 4),
            100 * exp(D[T] - P[T]), substr(s$Date[T], 1, 4)))
cat(sprintf("razón de pago D/E promedio: 1871-1950 = %.1f %%; 1990-2023 = %.1f %%\n",
            100 * mean(exp(D - E)[substr(s$Date, 1, 4) <= "1950"]), 100 * mean(exp(D - E)[substr(s$Date, 1, 4) >= "1990"])))
