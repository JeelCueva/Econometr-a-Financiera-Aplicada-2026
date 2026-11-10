# Capítulo 2, Ejercicio 2.4: el S&P 500 mensual de Shiller y el efecto del promedio (Working, 1960)
datos <- read.csv("datos/internacional/sp500_mensual_shiller.csv")
datos$Date <- as.Date(datos$Date)
r <- diff(log(datos$SP500))                # cambios logarítmicos mensuales del precio promedio
fechas <- datos$Date[-1]
cat("Período:", format(min(fechas)), "a", format(max(fechas)), " T =", length(r), "\n\n")

rho <- function(x, k = 6) acf(x, lag.max = k, plot = FALSE)$acf[2:(k + 1)]
tramos <- list(completo = rep(TRUE, length(r)),
               "1871-1945" = fechas <= as.Date("1945-12-01"),
               "1946-2026" = fechas >= as.Date("1946-01-01"))
for (n in names(tramos)) {
  x <- r[tramos[[n]]]
  cat(sprintf("%-10s T = %4d  banda = %.3f  rho(1..6) =", n, length(x), 1.96 / sqrt(length(x))),
      sprintf("%6.3f", rho(x)), "\n")
}
cat("\nValor teórico de Working para promedios de 21 días:", round((21^2 - 1) / (2 * (2 * 21^2 + 1)), 4), "\n")
lb <- Box.test(r, lag = 1, type = "Ljung-Box")
p  <- pchisq(lb$statistic, df = 1, lower.tail = FALSE)   # cola superior: evita 1 - F = 0 por redondeo
cat(sprintf("Ljung-Box (m = 1): Q = %.2f, p-valor = %.3g\n", lb$statistic, p))
