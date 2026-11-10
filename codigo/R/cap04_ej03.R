# Capítulo 4, Ejercicio 4.3: autocorrelaciones del Brent con bandas robustas
datos <- read.csv("datos/internacional/brent_diario.csv")
r <- diff(log(datos$Price)); r <- r - mean(r); T <- length(r)
res <- as.data.frame(t(sapply(1:10, function(j) {
  a <- r[(j + 1):T]; b <- r[1:(T - j)]
  rho <- sum(a * b) / sum(r^2)
  v   <- sum(a^2 * b^2) / sum(r^2)^2         # varianza robusta de rho_j
  c(rezago = j, rho = rho, banda_clasica = 1.96 / sqrt(T), banda_robusta = 1.96 * sqrt(v),
    t_clasico = rho * sqrt(T), t_robusto = rho / sqrt(v))
})))
print(round(res, 4), row.names = FALSE)
Q_lb  <- T * (T + 2) * sum(res$rho^2 / (T - 1:10))
Q_rob <- sum(res$t_robusto^2)
cat(sprintf("\nLjung-Box Q(10) = %.2f  p = %.4f\n", Q_lb, pchisq(Q_lb, 10, lower.tail = FALSE)))
cat(sprintf("Robusta  Q*(10) = %.2f  p = %.4f\n", Q_rob, pchisq(Q_rob, 10, lower.tail = FALSE)))
