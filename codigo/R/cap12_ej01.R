# Capítulo 12, Ejercicio 12.1: correlaciones que cambian: ventana móvil frente a EWMA
source("codigo/R/cap12_funciones.R")
P <- panel_diario(); R <- P$R; f <- P$fecha; T <- nrow(R)
cat(sprintf("Panel diario: %s a %s, T = %d días comunes al S&P 500, al Nasdaq y al Brent\n", f[1], f[T], T))
cat("Correlaciones de toda la muestra:\n"); print(round(cor(R), 4))
H <- ewma_cov(R); rho_e <- function(i, j) H[, i, j] / sqrt(H[, i, i] * H[, j, j])
mov <- function(i, j, n = 250) c(rep(NA, n), sapply((n + 1):T, function(t) cor(R[(t - n):(t - 1), i], R[(t - n):(t - 1), j])))
pares <- list("S&P-Brent" = c(1, 3), "S&P-Nasdaq" = c(1, 2))
for (nm in names(pares)) {
  ij <- pares[[nm]]; e <- rho_e(ij[1], ij[2]); m <- mov(ij[1], ij[2])
  cat(sprintf("\n%s: EWMA mín. %.3f (%s), máx. %.3f (%s); ventana de 250 días: mín. %.3f, máx. %.3f\n", nm,
              min(e[-(1:30)]), f[-(1:30)][which.min(e[-(1:30)])], max(e[-(1:30)]), f[-(1:30)][which.max(e[-(1:30)])],
              min(m, na.rm = TRUE), max(m, na.rm = TRUE)))
  for (per in list(c("1999", "2007"), c("2009", "2018"))) {
    k <- substr(f, 1, 4) >= per[1] & substr(f, 1, 4) <= per[2]
    cat(sprintf("  %s-%s: correlación muestral = %.3f\n", per[1], per[2], cor(R[k, ij[1]], R[k, ij[2]])))
  }
}
