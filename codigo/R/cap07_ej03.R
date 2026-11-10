# Capítulo 7, Ejercicio 7.3: reversión a la media de largo plazo (S&P, rendimiento real anual, Shiller)
source("codigo/R/cap07_funciones.R")
s <- read.csv("datos/internacional/sp500_mensual_shiller.csv")
s <- s[as.Date(s$Date) <= as.Date("2023-06-01"), ]
P <- s$SP500; D <- s$Dividend; I <- s$Consumer.Price.Index; n <- nrow(s)
g <- log((P[-1] + D[-1] / 12) / P[-n]) - diff(log(I))   # rendimiento real total mensual
idx <- c(0, cumsum(g))                                   # índice de riqueza real (log)
ene <- format(as.Date(s$Date), "%m") == "01"
ra <- diff(idx[ene]); T <- length(ra)
cat(sprintf("T = %d rendimientos reales anuales (enero a enero, %s a %s)\n",
            T, substr(s$Date[ene][1], 1, 4), substr(tail(s$Date[ene], 1), 1, 4)))
cat(sprintf("media = %.4f, desv. est. = %.4f, rho1 = %.3f\n", mean(ra), sd(ra), acf(ra, 1, plot = FALSE)$acf[2]))
cat("   q      VR        z       z*\n")
for (q in c(2, 3, 5, 8, 10)) {
  v <- vr_lm(ra, q); cat(sprintf("%4d %7.3f %8.2f %8.2f\n", q, v["VR"], v["z"], v["zrob"]))
}
