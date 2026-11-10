# Capítulo 13, Ejercicio 13.1: VaR y ES incondicionales a un día: normal, t de Student e histórico
source("codigo/R/cap13_funciones.R")
s <- read.csv("datos/internacional/sp500_diario_yahoo.csv"); b <- read.csv("datos/internacional/brent_diario.csv")
series <- list("S&P 500 (1999-2018)" = -100 * diff(log(s$Adj.Close)), "Brent (1987-2026)" = -100 * diff(log(b$Price)))
for (nm in names(series)) {
  L <- series[[nm]]; mu <- -mean(L); sg <- sd(L); e <- L - mean(L); k <- mean(e^4) / mean(e^2)^2
  nu <- 4 + 6 / (k - 3)                                   # t por momentos: curtosis = 3 + 6/(nu - 4)
  cat(sprintf("%s: T = %d, media de r = %.4f %%, desv. est. = %.4f %%, curtosis = %.2f -> nu = %.3f\n", nm, length(L), mu, sg, k, nu))
  cat("  nivel   modelo       VaR      ES   frecuencia de L > VaR\n")
  for (p in c(0.95, 0.99)) {
    res <- list(normal = var_es_normal(mu, sg, p), t = var_es_t(mu, sg, nu, p), "histórico" = var_es_hist(L, p))
    for (m in names(res)) cat(sprintf("  %4.0f %%   %-10s %6.3f  %6.3f   %6.3f %%\n", 100 * p, m, res[[m]]["VaR"], res[[m]]["ES"],
                                      100 * mean(L > res[[m]]["VaR"])))
  }
  cat("\n")
}
