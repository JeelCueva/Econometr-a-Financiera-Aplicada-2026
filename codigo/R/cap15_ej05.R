# Capítulo 15, Ejercicio 15.5: Vasicek y CKLS para la tasa a un mes
source("codigo/R/cap15_funciones.R")
D <- cargar(); r <- D$Y[, 1]; f <- D$fecha; n <- length(r); Delta <- 1 / 12
x <- r[-n]; y <- r[-1]
m <- lm(y ~ x); a <- coef(m)[1]; b <- coef(m)[2]; e <- resid(m); s <- sqrt(sum(e^2) / (n - 1 - 2))
cat(sprintf("AR(1) por MCO: r_t = a + b r_{t-1} + e_t;  n = %d\n", n - 1))
cat(sprintf("  a = %.5f (ee %.5f);  b = %.5f (ee %.5f);  s = %.5f\n", a, summary(m)$coef[1, 2], b, summary(m)$coef[2, 2], s))
kap <- -log(b) / Delta; th <- a / (1 - b); sig <- s * sqrt(2 * kap / (1 - b^2))
cat("Parámetros de Vasicek (Delta = 1/12 año):\n")
cat(sprintf("  kappa = %.4f por año;  theta = %.4f %%;  sigma = %.4f %% por raíz de año\n", kap, th, sig))
cat(sprintf("  vida media = ln 2 / kappa = %.3f años;  desv. est. estacionaria = sigma/sqrt(2 kappa) = %.4f %%\n", log(2) / kap, sig / sqrt(2 * kap)))
cat(sprintf("  media muestral = %.4f %%;  desv. est. muestral = %.4f %%;  P(r < 0) bajo Vasicek = %.4f\n", mean(r), sd(r), pnorm(0, th, sig / sqrt(2 * kap))))
cat("\nEfecto nivel (CKLS): ln e_t^2 = c + 2 gamma ln r_{t-1} + u_t\n")
mm <- lm(log(e^2) ~ log(x)); g2 <- coef(mm)[2]; eeg <- summary(mm)$coef[2, 2]
cat(sprintf("  2 gamma = %.4f (ee %.4f)  ->  gamma = %.4f;  t(gamma = 0) = %.2f;  t(gamma = 0,5) = %.2f\n", g2, eeg, g2 / 2, g2 / eeg, (g2 - 1) / eeg))
cat("\nDesv. est. de e_t por terciles de r_{t-1}:\n")
q <- quantile(x, c(1/3, 2/3)); g <- cut(x, c(-Inf, q, Inf), labels = c("bajo", "medio", "alto"))
for (k in levels(g)) cat(sprintf("  %-6s  r medio = %6.3f  desv.est.(e) = %.4f  n = %d\n", k, mean(x[g == k]), sd(e[g == k]), sum(g == k)))
