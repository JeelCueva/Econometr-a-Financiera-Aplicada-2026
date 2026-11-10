# Capítulo 17, Ejercicio 17.5: pronóstico de un mes y VaR con el modelo de dos regímenes
source("codigo/R/cap17_funciones.R")
D <- mercado(); f <- D$fecha; y <- D$y; T <- length(y); m <- mean(y); sd0 <- sqrt(mean((y - m)^2))
r <- em_media_varianza(y, 2, c(m, m), c(0.7, 1.5) * sd0, matrix(c(0.95, 0.05, 0.05, 0.95), 2, 2))
mu <- r$mu; s <- r$s; P <- r$P
w <- as.vector(r$xf[T, ] %*% P); mp <- sum(w * mu); vp <- sum(w * (s^2 + mu^2)) - mp^2
cat(sprintf("Pronóstico para el mes siguiente a %s:\n", f[T]))
cat(sprintf("  Pr filtrada del régimen volátil en %s = %.4f; Pr predicha = %.4f\n", f[T], r$xf[T, 2], w[2]))
cat(sprintf("  media = %.4f %%; desv. est. = %.4f %%\n", mp, sqrt(vp)))
for (a in c(0.05, 0.01)) cat(sprintf("  VaR al %.0f %%: mezcla = %.4f %%; normal incondicional = %.4f %%\n", 100 * (1 - a), -cuantil_mezcla(a, w, mu, s), -(m + sd0 * qnorm(a))))
W <- r$xf[-T, ] %*% P; y1 <- y[-1]; n <- length(y1)
cat(sprintf("\nPrueba retrospectiva de %d meses (%s a %s); parámetros estimados con toda la muestra\n", n, f[2], f[T]))
cat("  nivel  modelo        excepciones  esperadas   frecuencia   LR_uc   p_uc    LR_ind   p_ind\n")
for (a in c(0.05, 0.01)) {
  qm <- sapply(1:n, function(t) cuantil_mezcla(a, W[t, ], mu, s)); qn <- m + sd0 * qnorm(a)
  for (mod in c("dos regímenes", "normal      ")) {
    q <- if (mod == "dos regímenes") qm else rep(qn, n); h <- as.integer(y1 < q); k <- kupiec_christoffersen(h, a)
    cat(sprintf("  %5.2f  %s  %8d  %10.1f  %10.4f  %7.2f %6.4f  %7.2f %6.4f\n", a, mod, sum(h), a * n, mean(h), k[1], pchisq(k[1], 1, lower.tail = FALSE), k[2], pchisq(k[2], 1, lower.tail = FALSE)))
  }
  if (a == 0.01) Q1 <- qm
}
cat(sprintf("\nVaR al 99 %% de la mezcla: mínimo = %.4f %%; máximo = %.4f %%; mediana = %.4f %%\n", -max(Q1), -min(Q1), -median(Q1)))
