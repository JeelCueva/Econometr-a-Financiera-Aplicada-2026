# Capítulo 6, Ejercicio 6.2: estimación y diagnóstico de modelos ARMA para el logaritmo del VIX
v <- read.csv("datos/internacional/vix_diario.csv"); y <- log(v$CLOSE)
ordenes <- list("AR(1)" = c(1, 0, 0), "AR(2)" = c(2, 0, 0), "ARMA(1,1)" = c(1, 0, 1))
for (n in names(ordenes)) {
  m <- arima(y, order = ordenes[[n]], method = "ML"); k <- sum(ordenes[[n]][c(1, 3)])
  e <- residuals(m)[-1]                       # se excluye el primer residuo (depende de la inicialización)
  q10 <- Box.test(e, lag = 10, type = "Ljung-Box", fitdf = k)
  q20 <- Box.test(e, lag = 20, type = "Ljung-Box", fitdf = k)
  q10$p.value <- pchisq(q10$statistic, 10 - k, lower.tail = FALSE)   # cola superior (más precisa)
  q20$p.value <- pchisq(q20$statistic, 20 - k, lower.tail = FALSE)
  cat(sprintf("%-9s %s\n          ee: %s\n          AIC = %.2f  BIC = %.2f  Q(10) = %.1f (p = %.3g)  Q(20) = %.1f (p = %.3g)\n",
      n, paste(sprintf("%s=%.4f", names(coef(m)), coef(m)), collapse = " "),
      paste(sprintf("%.4f", sqrt(diag(m$var.coef))), collapse = " "),
      AIC(m), BIC(m), q10$statistic, q10$p.value, q20$statistic, q20$p.value))
}
m <- arima(y, order = c(1, 0, 1), method = "ML")
cat(sprintf("\nARMA(1,1): sigma2 = %.6f  raíz AR = %.4f  vida media = %.1f días\n",
            m$sigma2, 1 / coef(m)["ar1"], log(0.5) / log(coef(m)["ar1"])))
