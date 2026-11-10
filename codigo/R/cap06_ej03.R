# Capítulo 6, Ejercicio 6.3: ¿pronostica el ARMA(1,1) el VIX mejor que un paseo aleatorio? (fuera de muestra)
v <- read.csv("datos/internacional/vix_diario.csv"); y <- log(v$CLOSE); T <- length(y)
fin <- max(which(v$DATE <= "2015-12-31"))        # último día de la muestra de estimación
m <- arima(y[1:fin], order = c(1, 0, 1), method = "ML")
mu <- coef(m)["intercept"]; phi <- coef(m)["ar1"]; th <- coef(m)["ma1"]
cat(sprintf("Estimación %s a %s: mu = %.4f  phi = %.4f  theta = %.4f\n", v$DATE[1], v$DATE[fin], mu, phi, th))
# Residuos con los parámetros fijos: e_t = (y_t - mu) - phi (y_{t-1} - mu) - theta e_{t-1}
e <- numeric(T)
for (t in 2:T) e[t] <- (y[t] - mu) - phi * (y[t - 1] - mu) - th * e[t - 1]
media_est <- mean(y[1:fin])
dm <- function(d, h) {                               # Diebold-Mariano con varianza de Newey-West (L = h - 1)
  n <- length(d); u <- d - mean(d); S <- sum(u^2) / n
  if (h > 1) for (j in 1:(h - 1)) S <- S + 2 * (1 - j / h) * sum(u[(j + 1):n] * u[1:(n - j)]) / n
  mean(d) / sqrt(S / n)
}
cat(sprintf("Pronósticos desde %s; %d orígenes\n\n", v$DATE[fin], T - fin))
cat("   h   RMSE ARMA   RMSE paseo   RMSE media   ARMA/paseo   DM (paseo - ARMA)\n")
for (h in c(1, 5, 22)) {
  o <- fin:(T - h)                                   # orígenes de pronóstico
  f1 <- mu + phi * (y[o] - mu) + th * e[o]
  fa <- mu + phi^(h - 1) * (f1 - mu)                 # ARMA(1,1) a h pasos
  ea <- y[o + h] - fa; ep <- y[o + h] - y[o]; em <- y[o + h] - media_est
  r <- sqrt(c(mean(ea^2), mean(ep^2), mean(em^2)))
  cat(sprintf("%4d %11.4f %12.4f %12.4f %12.3f %16.2f\n", h, r[1], r[2], r[3], r[1] / r[2], dm(ep^2 - ea^2, h)))
}
