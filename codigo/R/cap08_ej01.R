# Capítulo 8, Ejercicio 8.1: ¿hay efectos ARCH? Ljung-Box sobre r y r^2 y prueba LM de Engle
b <- read.csv("datos/internacional/brent_diario.csv")
s <- read.csv("datos/internacional/sp500_mensual_shiller.csv")
series <- list("Brent diario" = 100 * diff(log(b$Price)), "S&P 500 mensual" = 100 * diff(log(s$SP500)))
arch_lm <- function(e, q) {                      # regresión de e_t^2 sobre q rezagos de e^2
  e2 <- e^2; n <- length(e2)
  X <- sapply(1:q, function(j) e2[(q + 1 - j):(n - j)])
  m <- lm(e2[(q + 1):n] ~ X); R2 <- summary(m)$r.squared
  c(LM = (n - q) * R2, p = pchisq((n - q) * R2, q, lower.tail = FALSE), R2 = R2)
}
for (nm in names(series)) {
  r <- series[[nm]]; e <- r - mean(r)
  q1 <- Box.test(e, 10, type = "Ljung-Box"); q2 <- Box.test(e^2, 10, type = "Ljung-Box")
  lm5 <- arch_lm(e, 5)
  cat(sprintf("%s (T = %d)\n", nm, length(r)))
  cat("  ACF de r   (1-5):", sprintf("%6.3f", acf(e, 5, plot = FALSE)$acf[2:6]), "\n")
  cat("  ACF de r^2 (1-5):", sprintf("%6.3f", acf(e^2, 5, plot = FALSE)$acf[2:6]), "\n")
  cat(sprintf("  Q(10) de r   = %8.2f  p = %.2e\n", q1$statistic, pchisq(q1$statistic, 10, lower.tail = FALSE)))
  cat(sprintf("  Q(10) de r^2 = %8.2f  p = %.2e\n", q2$statistic, pchisq(q2$statistic, 10, lower.tail = FALSE)))
  cat(sprintf("  LM de Engle (5 rezagos): R2 = %.4f, LM = %.2f, p = %.2e\n\n", lm5["R2"], lm5["LM"], lm5["p"]))
}
