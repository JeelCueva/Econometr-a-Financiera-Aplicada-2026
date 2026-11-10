# Capítulo 7, Ejercicio 7.5: el factor momentum como anomalía de la eficiencia débil
f <- read.csv("datos/internacional/french_mensual_1949_2017.csv")
m <- f$Mom; T <- length(m)
L <- floor(4 * (T / 100)^(2 / 9))                         # regla de Newey-West
e <- m - mean(m); s <- sum(e^2) / T
for (j in 1:L) s <- s + 2 * (1 - j / (L + 1)) * sum(e[(j + 1):T] * e[1:(T - j)]) / T
cat(sprintf("T = %d, media mensual = %.4f (anual %.2f %%), desv. est. anual = %.2f %%\n",
            T, mean(m), 1200 * mean(m), 100 * sd(m) * sqrt(12)))
cat(sprintf("t clásico = %.2f; t Newey-West (L = %d) = %.2f\n",
            mean(m) / (sd(m) / sqrt(T)), L, mean(m) / sqrt(s / T)))
cat(sprintf("razón de Sharpe anual = %.3f; asimetría = %.3f\n",
            sqrt(12) * mean(m) / sd(m), mean(e^3) / mean(e^2)^1.5))
w <- cumsum(log(1 + m)); dd <- w - cummax(w)
cat(sprintf("peor mes: %s (%.2f %%); caída máxima acumulada = %.1f %% (fondo en %s)\n",
            f$dates[which.min(m)], 100 * min(m), 100 * (exp(min(dd)) - 1), f$dates[which.min(dd)]))
sub <- list("1949-1999" = f$dates <= "1999-12", "2000-2017" = f$dates >= "2000-01")
for (n in names(sub)) {
  x <- m[sub[[n]]]
  cat(sprintf("  %s: media anual = %6.2f %%, t = %5.2f\n", n, 1200 * mean(x), mean(x) / (sd(x) / sqrt(length(x)))))
}
