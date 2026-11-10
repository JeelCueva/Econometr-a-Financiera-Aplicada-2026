# Capítulo 2, Ejercicio 2.2: autocorrelación de r, r^2 y |r| (Brent diario)
datos <- read.csv("datos/internacional/brent_diario.csv")
r <- diff(log(datos$Price))
T <- length(r)
series <- list(r = r, r2 = r^2, abs_r = abs(r))

cat("Banda de Bartlett al 95 %: +/-", round(1.96 / sqrt(T), 4), "\n\n")
tabla <- sapply(series, function(x) acf(x, lag.max = 10, plot = FALSE)$acf[2:11])
rownames(tabla) <- paste0("rezago_", 1:10)
print(round(tabla, 4))

cat("\nPrueba de Ljung-Box (H0: sin autocorrelación hasta el rezago m)\n")
for (nombre in names(series)) for (m in c(10, 20)) {
  lb <- Box.test(series[[nombre]], lag = m, type = "Ljung-Box")
  p  <- pchisq(lb$statistic, df = m, lower.tail = FALSE)  # cola superior, más precisa que lb$p.value
  cat(sprintf("%-6s m = %2d  Q = %10.2f  p-valor = %.4g\n", nombre, m, lb$statistic, p))
}
