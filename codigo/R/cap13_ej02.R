# Capítulo 13, Ejercicio 13.2: prueba retrospectiva del VaR al 99 % del S&P 500 (2001-2018)
source("codigo/R/cap13_funciones.R"); source("codigo/R/cap08_funciones.R")
s <- read.csv("datos/internacional/sp500_diario_yahoo.csv"); r <- 100 * diff(log(s$Adj.Close)); L <- -r
f <- s$Date[-1]; anio <- as.integer(substr(f, 1, 4)); T <- length(r); p <- 0.99; ev <- which(anio >= 2001)
V <- matrix(NA, T, 3, dimnames = list(NULL, c("HS-500", "EWMA-normal", "GARCH-t")))
for (t in ev) V[t, 1] <- unname(quantile(L[(t - 500):(t - 1)], p, type = 7))   # simulación histórica, 500 días
s2 <- numeric(T); s2[1] <- var(r[1:30])
for (t in 2:T) s2[t] <- 0.94 * s2[t - 1] + 0.06 * r[t - 1]^2                  # RiskMetrics, media cero
V[ev, 2] <- qnorm(p) * sqrt(s2[ev])
for (y in 2001:2018) {                                                         # GARCH-t reestimado cada enero
  m <- garch_fit(r[anio < y], "t"); q <- m$par; h <- garch_s2(q, r, m$bc); k <- which(anio == y)
  V[k, 3] <- -q["mu"] + sqrt(h[k]) * sqrt((q["nu"] - 2) / q["nu"]) * qt(p, q["nu"])
}
cat(sprintf("Evaluación: %s a %s, n = %d días; excepciones esperadas al 1 %%: %.1f\n\n", f[ev[1]], f[T], length(ev), 0.01 * length(ev)))
cat("Modelo         VaR medio  excepciones  tasa (%)  Kupiec LR (p)     Christoffersen LR (p)   cobertura condicional (p)\n")
for (j in 1:3) {
  x <- as.numeric(L[ev] > V[ev, j]); ku <- kupiec(x, p); ch <- christoffersen(x); cc <- ku["LR"] + ch["LR"]
  cat(sprintf("%-12s %9.3f %10d %10.3f %8.2f (%.4f) %10.2f (%.4f) %14.2f (%.4f)\n", colnames(V)[j], mean(V[ev, j]), sum(x), 100 * mean(x),
              ku["LR"], ku["p"], ch["LR"], ch["p"], cc, pchisq(cc, 2, lower.tail = FALSE)))
}
k <- anio[ev] == 2008
cat(sprintf("\nexcepciones en 2008: HS-500 = %d; EWMA = %d; GARCH-t = %d (de %d días)\n",
            sum(L[ev][k] > V[ev, 1][k]), sum(L[ev][k] > V[ev, 2][k]), sum(L[ev][k] > V[ev, 3][k]), sum(k)))
write.csv(data.frame(fecha = f[ev], L = L[ev], V[ev, ], check.names = FALSE), "salidas/cap13_var_backtest_R.csv", row.names = FALSE)
