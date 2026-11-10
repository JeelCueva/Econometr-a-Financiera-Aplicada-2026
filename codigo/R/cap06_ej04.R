# Capítulo 6, Ejercicio 6.4: pronóstico del VIX hacia adelante con intervalos de confianza
v <- read.csv("datos/internacional/vix_diario.csv"); y <- log(v$CLOSE); T <- length(y)
m <- arima(y, order = c(1, 0, 1), method = "ML")
mu <- coef(m)["intercept"]; phi <- coef(m)["ar1"]; th <- coef(m)["ma1"]; s2 <- m$sigma2
e <- numeric(T); for (t in 2:T) e[t] <- (y[t] - mu) - phi * (y[t - 1] - mu) - th * e[t - 1]
H <- 60; h <- 1:H
f1 <- mu + phi * (y[T] - mu) + th * e[T]
f  <- mu + phi^(h - 1) * (f1 - mu)
psi <- c(1, (phi + th) * phi^(0:(H - 2)))           # pesos psi del MA(infinito)
ecm <- s2 * cumsum(psi^2)                            # error cuadrático medio a h pasos
cat(sprintf("Origen: %s, VIX = %.2f\n", v$DATE[T], exp(y[T])))
cat("   h   mediana   IC 95 %\n")
for (k in c(1, 5, 22, 60))
  cat(sprintf("%4d %9.2f   [%5.2f; %5.2f]\n", k, exp(f[k]), exp(f[k] - 1.96 * sqrt(ecm[k])), exp(f[k] + 1.96 * sqrt(ecm[k]))))
cat(sprintf("Límite de largo plazo: mediana %.2f, IC 95 %% [%.2f; %.2f]\n", exp(mu),
            exp(mu - 1.96 * sqrt(s2 * (1 + (phi + th)^2 / (1 - phi^2)))), exp(mu + 1.96 * sqrt(s2 * (1 + (phi + th)^2 / (1 - phi^2))))))
