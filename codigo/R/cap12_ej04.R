# Capítulo 12, Ejercicio 12.4: cartera de mínima varianza fuera de muestra con covarianzas constante, EWMA y DCC
source("codigo/R/cap12_funciones.R")
P <- panel_diario(); R <- P$R; f <- P$fecha; T <- nrow(R); K <- 3
dentro <- f <= "2009-12-31"; fuera <- which(!dentro); n0 <- sum(dentro)
e1 <- etapa1(R[dentro, ]); m <- dcc_fit(e1$Z)                     # todo se estima con datos hasta 2009
S2 <- Z <- matrix(0, T, K)
for (i in 1:K) { mo <- e1$modelos[[i]]; S2[, i] <- garch_s2(mo$par, R[, i], mo$bc); Z[, i] <- (R[, i] - mo$par["mu"]) / sqrt(S2[, i]) }
Rt <- dcc_filtro(c(m$a, m$b), Z, Qb = crossprod(e1$Z) / n0)        # mismos a, b y Qbar de la muestra de estimación
He <- ewma_cov(R); Sc <- cov(R[dentro, ])
pesos <- function(H) { x <- solve(H, rep(1, K)); x / sum(x) }
W <- list(constante = matrix(0, T, K), EWMA = matrix(0, T, K), DCC = matrix(0, T, K), iguales = matrix(1 / 3, T, K))
for (t in fuera) {
  D <- diag(sqrt(S2[t, ])); W$constante[t, ] <- pesos(Sc); W$EWMA[t, ] <- pesos(He[t, , ]); W$DCC[t, ] <- pesos(D %*% Rt[t, , ] %*% D)
}
rp <- sapply(W, function(w) rowSums(w[fuera, ] * R[fuera, ]))
cat(sprintf("Estimación: %s a %s (%d días); evaluación: %s a %s (%d días)\n", f[1], f[n0], n0, f[fuera[1]], f[T], length(fuera)))
cat(sprintf("DCC estimado: a = %.4f, b = %.4f\n\n", m$a, m$b))
cat("Estrategia     volat. anual (%)   peso medio S&P  Nasdaq   Brent   rotación diaria media\n")
for (nm in names(W)) {
  w <- W[[nm]][fuera, ]; rot <- mean(rowSums(abs(diff(w))))
  cat(sprintf("%-12s %12.3f %15.3f %8.3f %7.3f %12.4f\n", nm, sqrt(252) * sd(rp[, nm]), mean(w[, 1]), mean(w[, 2]), mean(w[, 3]), rot))
}
nw_t <- function(d) { n <- length(d); L <- floor(4 * (n / 100)^(2 / 9)); u <- d - mean(d); s <- sum(u^2) / n
  for (j in 1:L) s <- s + 2 * (1 - j / (L + 1)) * sum(u[(j + 1):n] * u[1:(n - j)]) / n; mean(d) / sqrt(s / n) }
cat("\nDiebold-Mariano sobre r_p^2 (t > 0: la DCC tiene menor varianza):\n")
for (nm in c("constante", "EWMA", "iguales")) {
  d <- rp[, nm]^2 - rp[, "DCC"]^2; tt <- nw_t(d)
  cat(sprintf("  DCC frente a %-10s t = %5.2f (p bilateral = %.4f)\n", nm, tt, 2 * pnorm(-abs(tt))))
}
