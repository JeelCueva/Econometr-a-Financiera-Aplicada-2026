# Capítulo 13, Ejercicio 13.4: teoría de valores extremos para las pérdidas del Brent y del S&P 500
source("codigo/R/cap13_funciones.R")
b <- read.csv("datos/internacional/brent_diario.csv"); s <- read.csv("datos/internacional/sp500_diario_yahoo.csv")
series <- list(Brent = -100 * diff(log(b$Price)), "S&P 500" = -100 * diff(log(s$Adj.Close)))
for (nm in names(series)) {
  L <- series[[nm]]; n <- length(L)
  cat(sprintf("%s: n = %d pérdidas diarias (%%)\n", nm, n))
  cat("  umbral (cuantil)      u     excesos      xi (ee)            beta (ee)\n")
  for (q in c(0.90, 0.95, 0.975)) {
    u <- unname(quantile(L, q, type = 7)); y <- L[L > u] - u; g <- gpd_fit(y)
    cat(sprintf("  %5.1f %%           %6.3f   %5d     %6.4f (%.4f)   %6.4f (%.4f)\n", 100 * q, u, length(y), g$xi, g$ee[1], g$beta, g$ee[2]))
  }
  u <- unname(quantile(L, 0.95, type = 7)); y <- L[L > u] - u; g <- gpd_fit(y)
  mu <- -mean(L); sg <- sd(L); e <- L - mean(L); nu <- 4 + 6 / (mean(e^4) / mean(e^2)^2 - 3)
  cat("  con u en el cuantil 95 %:\n  nivel      GPD: VaR    ES     normal: VaR    ES     t: VaR    ES     histórico: VaR    ES\n")
  for (p in c(0.99, 0.995, 0.999)) {
    a <- gpd_var_es(u, g$xi, g$beta, n, length(y), p); bn <- var_es_normal(mu, sg, p); bt <- var_es_t(mu, sg, nu, p); h <- var_es_hist(L, p)
    cat(sprintf("  %5.1f %%   %8.3f %6.3f   %10.3f %6.3f   %7.3f %6.3f   %12.3f %6.3f\n", 100 * p, a[1], a[2], bn[1], bn[2], bt[1], bt[2], h[1], h[2]))
  }
  cat("\n")
}
