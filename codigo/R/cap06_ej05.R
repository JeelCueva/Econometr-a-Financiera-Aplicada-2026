# Capítulo 6, Ejercicio 6.5: el sesgo de Stambaugh en la regresión predictiva con la razón dividendo/precio
s <- read.csv("datos/internacional/sp500_mensual_shiller.csv")
s <- s[as.Date(s$Date) <= as.Date("2023-06-01"), ]
P <- s$SP500; D <- s$Dividend; n <- length(P)
r  <- log((P[-1] + D[-1] / 12) / P[-n])             # rendimiento total del mes siguiente
dp <- log(D / P)
usar <- as.Date(s$Date[-n]) >= as.Date("1946-01-01")
x <- dp[-n][usar]; x1 <- dp[-1][usar]; r <- r[usar]; T <- length(r)
m1 <- lm(r ~ x); m2 <- lm(x1 ~ x)                   # regresión predictiva y AR(1) del predictor
b <- coef(m1)[2]; phi <- coef(m2)[2]
u <- resid(m1); v <- resid(m2)
k <- cov(u, v) / var(v)                             # pendiente de u sobre v
sesgo_phi <- -(1 + 3 * phi) / T                      # Kendall
sesgo_b <- k * sesgo_phi                             # Stambaugh
cat(sprintf("T = %d meses (%s a %s)\n", T, s$Date[which(usar)[1]], s$Date[max(which(usar))]))
cat(sprintf("b = %.5f (t = %.2f)   phi = %.5f   corr(u, v) = %.3f   sigma_uv/sigma_v^2 = %.4f\n",
            b, summary(m1)$coefficients[2, 3], phi, cor(u, v), k))
cat(sprintf("Sesgo de Kendall de phi = %.5f\nSesgo de Stambaugh de b = %.5f\nb corregido = %.5f\n",
            sesgo_phi, sesgo_b, b - sesgo_b))
cat(sprintf("Proporción del coeficiente atribuible al sesgo = %.1f %%\n", 100 * sesgo_b / b))
