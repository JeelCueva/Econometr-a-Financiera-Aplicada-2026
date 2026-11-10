# Capítulo 7, Ejercicio 7.1: razón de varianzas del Brent diario
source("codigo/R/cap07_funciones.R")
b <- read.csv("datos/internacional/brent_diario.csv")
r <- diff(log(b$Price)); T <- length(r)
cat(sprintf("T = %d rendimientos diarios (%s a %s)\n", T, b$Date[2], b$Date[nrow(b)]))
cat("ACF 1-5:", sprintf("%6.3f", acf(r, 5, plot = FALSE)$acf[2:6]), "\n")
cat("   q       VR        z       z*\n")
for (q in c(2, 5, 10, 20)) {
  v <- vr_lm(r, q)
  cat(sprintf("%4d %8.4f %8.2f %8.2f\n", q, v["VR"], v["z"], v["zrob"]))
}
