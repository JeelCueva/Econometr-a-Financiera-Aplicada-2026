# Capítulo 16: funciones propias para estudios de eventos
ACT <- c("AAPL", "AMZN", "FB", "GOOG")
cargar <- function() {                                     # rendimientos simples diarios (%)
  d <- read.csv("datos/internacional/gafa_sp500_diario_2014_2018.csv"); P <- as.matrix(d[, c(ACT, "SP500")])
  list(fecha = d$fecha[-1], R = 100 * (P[-1, ] / P[-nrow(P), ] - 1))
}
rendimientos_normales <- function(r, m, t0, est = c(-250, -11), evt = c(-5, 5), modelo = "mercado", excluir = integer(0)) {
  ie <- setdiff((t0 + est[1]):(t0 + est[2]), excluir); iv <- (t0 + evt[1]):(t0 + evt[2])
  y <- if (modelo == "ajustado") r - m else r
  if (modelo == "mercado") { X <- cbind(1, m[ie]); Xs <- cbind(1, m[iv]) } else { X <- matrix(1, length(ie), 1); Xs <- matrix(1, length(iv), 1) }
  if (modelo == "ajustado") { b <- 0; e <- y[ie]; k <- 0; AR <- y[iv] } else {
    b <- solve(crossprod(X), crossprod(X, y[ie])); e <- as.vector(y[ie] - X %*% b); k <- ncol(X); AR <- as.vector(y[iv] - Xs %*% b) }
  s2 <- sum(e^2) / (length(ie) - k)
  V <- s2 * (diag(length(iv)) + if (k > 0) Xs %*% solve(crossprod(X), t(Xs)) else 0)
  list(AR = AR, V = V, s2 = s2, b = as.vector(b), gl = length(ie) - k, tau = evt[1]:evt[2], e = e)
}
car <- function(res, t1, t2) {                             # CAR(t1, t2) y su desviación estándar
  g <- as.numeric(res$tau >= t1 & res$tau <= t2); c(sum(g * res$AR), sqrt(drop(t(g) %*% res$V %*% g)))
}
pruebas <- function(AR, V, gl) {                           # J1, Patell, BMP y signo para un día
  N <- length(AR); SAR <- AR / sqrt(V)
  list(CAAR = mean(AR), J1 = mean(AR) / sqrt(sum(V) / N^2), Patell = sum(SAR) / sqrt(sum(gl / (gl - 2))),
       BMP = mean(SAR) / (sd(SAR) / sqrt(N)), signo = (sum(AR > 0) - N / 2) / sqrt(N / 4), pos = sum(AR > 0), SAR = SAR)
}
corrado <- function(A) {                                   # rangos de Corrado; última columna = día 0
  L <- ncol(A); K <- t(apply(A, 1, rank)); D <- colMeans(K - (L + 1) / 2)
  D[L] / sqrt(mean(D^2))
}
lcg_nuevo <- function(semilla) { e <- new.env(); e$x <- semilla; e }   # Park y Miller (1988)
lcg_u <- function(g) { g$x <- (16807 * g$x) %% 2147483647; g$x / 2147483647 }
