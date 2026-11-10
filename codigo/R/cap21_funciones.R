# Capítulo 21: funciones del proyecto integrador (reutiliza los Capítulos 8 y 13)
source("codigo/R/cap08_funciones.R"); source("codigo/R/cap13_funciones.R")
NOMBRES <- c("AAPL", "AMZN", "FB", "GOOG", "SP500")
cargar <- function() { d <- read.csv("datos/internacional/gafa_sp500_diario_2014_2018.csv"); P <- as.matrix(d[, NOMBRES])
  list(fecha = d$fecha[-1], R = 100 * (P[-1, ] / P[-nrow(P), ] - 1)) }
asimetria <- function(x) { c <- x - mean(x); mean(c^3) / mean(c^2)^1.5 }
curtosis <- function(x) { c <- x - mean(x); mean(c^4) / mean(c^2)^2 }
ljung_box <- function(x, m) { n <- length(x); c <- x - mean(x); g0 <- sum(c^2)
  r <- sapply(1:m, function(k) sum(c[(k + 1):n] * c[1:(n - k)]) / g0); Q <- n * (n + 2) * sum(r^2 / (n - 1:m)); c(Q, pchisq(Q, m, lower.tail = FALSE)) }
nw_mco <- function(y, X, L) {
  b <- solve(crossprod(X), crossprod(X, y)); e <- as.vector(y - X %*% b); g <- X * e; S <- crossprod(g); n <- nrow(X)
  for (j in 1:L) { G <- t(g[(j + 1):n, , drop = FALSE]) %*% g[1:(n - j), , drop = FALSE]; S <- S + (1 - j / (L + 1)) * (G + t(G)) }
  A <- solve(crossprod(X)); list(b = as.vector(b), ee = sqrt(diag(A %*% S %*% A)), r2 = 1 - sum(e^2) / sum((y - mean(y))^2))
}
ewma_var <- function(x, lam = 0.94, n0 = 30) { h <- numeric(length(x)); h[1] <- mean(x[1:n0]^2)
  for (i in 2:length(x)) h[i] <- lam * h[i - 1] + (1 - lam) * x[i - 1]^2; h }
min_varianza <- function(S) { w <- solve(S, rep(1, nrow(S))); w / sum(w) }
max_caida <- function(r) { v <- cumprod(1 + r / 100); 100 * (1 - min(v / cummax(v))) }
