# Capítulo 13, Ejercicio 13.3: el semáforo de Basilea, año por año
d <- read.csv("salidas/cap13_var_backtest_R.csv", check.names = FALSE); anio <- as.integer(substr(d$fecha, 1, 4))
cat("Probabilidad binomial de k o menos excepciones en 250 días si la cobertura es correcta (p = 0,01):\n")
for (k in c(4, 9)) cat(sprintf("  P(X <= %d) = %.4f\n", k, pbinom(k, 250, 0.01)))
zona <- function(k) ifelse(k <= 4, "verde", ifelse(k <= 9, "amarilla", "roja"))
extra <- c(0, 0, 0, 0, 0, 0.40, 0.50, 0.65, 0.75, 0.85)                    # factor adicional para k = 0..9
mult <- function(k) 3 + ifelse(k >= 10, 1, extra[pmin(k, 9) + 1])
cat("\nAño   días   HS-500        EWMA-normal   GARCH-t\n")
tot <- c(verde = 0, amarilla = 0, roja = 0); tz <- list()
for (y in sort(unique(anio))) {
  k <- anio == y; ex <- sapply(c("HS-500", "EWMA-normal", "GARCH-t"), function(m) sum(d$L[k] > d[[m]][k]))
  cat(sprintf("%d  %4d   %2d %-9s  %2d %-9s  %2d %-9s\n", y, sum(k), ex[1], zona(ex[1]), ex[2], zona(ex[2]), ex[3], zona(ex[3])))
  tz[[as.character(y)]] <- ex
}
Z <- do.call(rbind, tz)
for (m in colnames(Z)) cat(sprintf("%-12s años en verde %2d, amarilla %2d, roja %2d; multiplicador medio %.3f\n", m,
                                   sum(Z[, m] <= 4), sum(Z[, m] >= 5 & Z[, m] <= 9), sum(Z[, m] >= 10), mean(mult(Z[, m]))))
