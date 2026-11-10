# Capítulo 6, Ejercicio 6.1: identificación de un ARMA para los cambios de la tasa a 10 años
s <- read.csv("datos/internacional/sp500_mensual_shiller.csv")
s <- s[as.Date(s$Date) >= as.Date("1953-01-01") & as.Date(s$Date) <= as.Date("2023-06-01"), ]
di <- diff(s$Long.Interest.Rate); T <- length(di)
cat(sprintf("T = %d cambios mensuales; banda 95 %% = +/- %.3f\n", T, 1.96 / sqrt(T)))
cat("ACF :", sprintf("%6.3f", acf(di, 6, plot = FALSE)$acf[2:7]), "\n")
cat("PACF:", sprintf("%6.3f", pacf(di, 6, plot = FALSE)$acf[1:6]), "\n\n")
ordenes <- list("AR(1)" = c(1, 0, 0), "AR(2)" = c(2, 0, 0), "MA(1)" = c(0, 0, 1),
                "MA(2)" = c(0, 0, 2), "ARMA(1,1)" = c(1, 0, 1))
for (n in names(ordenes)) {
  m <- arima(di, order = ordenes[[n]], method = "ML")
  cf <- coef(m); cf <- cf[names(cf) != "intercept"]
  cat(sprintf("%-10s %-32s AIC = %7.2f  BIC = %7.2f\n", n,
              paste(sprintf("%s=%.4f", names(cf), cf), collapse = " "), AIC(m), BIC(m)))
}
cat(sprintf("\nMA(1) implicado por el efecto de Working (rho1 = 1/4): theta = 2 - sqrt(3) = %.4f\n", 2 - sqrt(3)))
