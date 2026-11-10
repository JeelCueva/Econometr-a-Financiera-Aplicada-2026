# Capítulo 9, Ejercicio 9.1: evidencia de asimetría (correlaciones cruzadas y pruebas de sesgo de signo de Engle-Ng)
source("codigo/R/cap09_funciones.R")
s <- read.csv("datos/internacional/sp500_diario_yahoo.csv")
b <- read.csv("datos/internacional/brent_diario.csv")
series <- list("S&P 500 (1999-2018)" = 100 * diff(log(s$Adj.Close)), "Brent (1987-2026)" = 100 * diff(log(b$Price)))
for (nm in names(series)) {
  r <- series[[nm]]; e <- r - mean(r); n <- length(e)
  cc <- function(x, y, j) cor(x[1:(n - j)], y[(1 + j):n])          # corr(x_{t-j}, y_t)
  cat(sprintf("%s, T = %d\n", nm, n))
  cat("  corr(r_{t-j}, r_t^2), j = 1..5:", sprintf("%7.3f", sapply(1:5, function(j) cc(e, e^2, j))), "\n")
  cat("  corr(r_{t-j}^2, r_t), j = 1..5:", sprintf("%7.3f", sapply(1:5, function(j) cc(e^2, e, j))), "\n")
  g <- ajustar(r, "garch", "c", "normal"); z <- g$e / sqrt(g$s2)  # residuos del GARCH(1,1) simétrico
  y <- z[-1]^2; zl <- z[-n]; Sm <- as.numeric(zl < 0)
  m <- lm(y ~ Sm + I(Sm * zl) + I((1 - Sm) * zl))
  ct <- summary(m)$coefficients
  cat(sprintf("  Engle-Ng sobre z_t^2 del GARCH(1,1):\n"))
  cat(sprintf("    %-22s %8.4f (t = %6.2f)\n", c("constante", "sesgo de signo S-", "tamaño negativo S- z", "tamaño positivo S+ z"),
              ct[, 1], ct[, 3]), sep = "")
  LM <- (n - 1) * summary(m)$r.squared
  cat(sprintf("    prueba conjunta: LM = (T-1) R2 = %.2f, p = %.2e\n\n", LM, pchisq(LM, 3, lower.tail = FALSE)))
}
