# Capítulo 10, Ejercicio 10.5: ¿pronostica el VAR mejor que la media? Evaluación fuera de muestra (cartera pequeña)
f <- read.csv("datos/internacional/french_mensual_1949_2017.csv")
y <- 100 * f$S1V3; x <- 100 * f$S5V3; T <- length(y)
t0 <- which(f$dates == "1983-01")                         # primer mes pronosticado
pm <- pa <- pv <- numeric(0)
for (t in t0:T) {                                          # ventana creciente: datos hasta t-1
  i <- 2:(t - 1)
  pm <- c(pm, mean(y[1:(t - 1)]))
  ba <- coef(lm(y[i] ~ y[i - 1]));            pa <- c(pa, ba[1] + ba[2] * y[t - 1])
  bv <- coef(lm(y[i] ~ y[i - 1] + x[i - 1])); pv <- c(pv, bv[1] + bv[2] * y[t - 1] + bv[3] * x[t - 1])
}
real <- y[t0:T]; em <- real - pm; ea <- real - pa; ev <- real - pv
cat(sprintf("Pronósticos a un mes: %s a %s (%d meses); estimación inicial %s a %s\n",
            f$dates[t0], f$dates[T], length(real), f$dates[1], f$dates[t0 - 1]))
r2 <- function(e) 100 * (1 - sum(e^2) / sum(em^2))
cat(sprintf("ECM: media = %.4f; AR(1) = %.4f; VAR(1) = %.4f\n", mean(em^2), mean(ea^2), mean(ev^2)))
cat(sprintf("R2 fuera de muestra respecto de la media: AR(1) = %.2f %%; VAR(1) = %.2f %%\n", r2(ea), r2(ev)))
clark_west <- function(e1, e2, p1, p2) {                  # modelo 1 anidado en el 2; H0: igual precisión
  fcw <- e1^2 - (e2^2 - (p1 - p2)^2); n <- length(fcw); u <- fcw - mean(fcw)
  L <- floor(4 * (n / 100)^(2 / 9)); s <- sum(u^2) / n
  for (j in 1:L) s <- s + 2 * (1 - j / (L + 1)) * sum(u[(j + 1):n] * u[1:(n - j)]) / n
  tt <- mean(fcw) / sqrt(s / n); c(t = tt, p = pnorm(tt, lower.tail = FALSE))
}
cw1 <- clark_west(em, ea, pm, pa); cw2 <- clark_west(em, ev, pm, pv); cw3 <- clark_west(ea, ev, pa, pv)
cat(sprintf("Clark-West (unilateral): media vs AR(1): t = %.2f (p = %.4f)\n", cw1["t"], cw1["p"]))
cat(sprintf("                         media vs VAR(1): t = %.2f (p = %.4f)\n", cw2["t"], cw2["p"]))
cat(sprintf("                         AR(1) vs VAR(1): t = %.2f (p = %.4f)\n", cw3["t"], cw3["p"]))
for (per in list(c("1983-01", "1999-12"), c("2000-01", "2017-12"))) {
  k <- f$dates[t0:T] >= per[1] & f$dates[t0:T] <= per[2]
  cat(sprintf("  %s a %s: R2 fuera de muestra AR(1) = %6.2f %%; VAR(1) = %6.2f %%\n", per[1], per[2],
              100 * (1 - sum(ea[k]^2) / sum(em[k]^2)), 100 * (1 - sum(ev[k]^2) / sum(em[k]^2))))
}
