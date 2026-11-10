# Capítulo 7, Ejercicio 7.2: portafolios pequeños frente a grandes; efecto adelanto-rezago
source("codigo/R/cap07_funciones.R")
f <- read.csv("datos/internacional/french_mensual_1949_2017.csv")
cat(sprintf("Muestra: %s a %s, T = %d meses\n\n", f$dates[1], f$dates[nrow(f)], nrow(f)))
cat("Cartera    rho1   VR(2)  z*(2)   VR(4)  z*(4)  VR(12) z*(12)\n")
for (p in c("S1V1", "S1V3", "S1V5", "S3V3", "S5V1", "S5V3", "S5V5")) {
  x <- f[[p]]; rho1 <- acf(x, 1, plot = FALSE)$acf[2]
  v <- sapply(c(2, 4, 12), function(q) vr_lm(x, q)[c("VR", "zrob")])
  cat(sprintf("%-8s %6.3f  %s\n", p, rho1, paste(sprintf("%6.2f %6.2f", v[1, ], v[2, ]), collapse = " ")))
}
pe <- f$S1V3; gr <- f$S5V3; n <- length(pe)
cat("\nCorrelaciones cruzadas (pequeña = S1V3, grande = S5V3):\n")
for (k in 0:3) {
  a <- cor(pe[(k + 1):n], gr[1:(n - k)])                 # pequeña_t con grande_{t-k}
  b <- cor(gr[(k + 1):n], pe[1:(n - k)])                 # grande_t con pequeña_{t-k}
  cat(sprintf("  k = %d: corr(peq_t, gran_t-k) = %6.3f   corr(gran_t, peq_t-k) = %6.3f\n", k, a, b))
}
m <- lm(pe[2:n] ~ pe[1:(n - 1)] + gr[1:(n - 1)])
cat("\nRegresión peq_t = a + b1 peq_t-1 + b2 gran_t-1:\n"); print(round(summary(m)$coefficients, 4))
