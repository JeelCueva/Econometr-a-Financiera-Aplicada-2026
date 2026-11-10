# Capítulo 10, Ejercicio 10.2: causalidad de Granger (S&P 500 y VIX diarios; carteras pequeña y grande mensuales)
source("codigo/R/cap10_funciones.R")
s <- read.csv("datos/internacional/sp500_diario_yahoo.csv"); v <- read.csv("datos/internacional/vix_diario.csv")
d <- merge(s[, c("Date", "Adj.Close")], v[, c("DATE", "CLOSE")], by.x = "Date", by.y = "DATE")
Y1 <- cbind(r = 100 * diff(log(d$Adj.Close)), dv = 100 * diff(log(d$CLOSE)))
f <- read.csv("datos/internacional/french_mensual_1949_2017.csv"); Y2 <- 100 * cbind(peq = f$S1V3, gran = f$S5V3)
reporta <- function(Y, p, efecto, causa, et) {
  g <- granger(Y, p, efecto, causa)
  cat(sprintf("  %-34s p = %d: F(%d, %d) = %6.3f (p = %.4f); Wald HC0 = %6.3f (p = %.4f)\n",
              et, p, g["gl1"], g["gl2"], g["F"], g["pF"], g["W"], g["pW"]))
}
cat("S&P 500 y VIX diarios, 1999-2018:\n")
for (p in c(2, 7)) { reporta(Y1, p, 1, 2, "dv no causa a r:"); reporta(Y1, p, 2, 1, "r no causa a dv:") }
cat("\nCarteras S1V3 (pequeña) y S5V3 (grande), mensuales 1949-2017:\n")
ic <- seleccion(Y2, 6); cat(sprintf("  orden elegido: AIC = %d, BIC = %d\n", ic[which.min(ic[, 2]), 1], ic[which.min(ic[, 3]), 1]))
for (p in c(1, 5)) { reporta(Y2, p, 1, 2, "la grande no causa a la pequeña:"); reporta(Y2, p, 2, 1, "la pequeña no causa a la grande:") }
