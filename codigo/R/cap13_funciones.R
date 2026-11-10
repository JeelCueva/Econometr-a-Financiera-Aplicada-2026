# Capítulo 13: funciones propias para VaR, ES, pruebas retrospectivas y teoría de valores extremos
var_es_normal <- function(mu, s, p) { z <- qnorm(p); c(VaR = -mu + s * z, ES = -mu + s * dnorm(z) / (1 - p)) }
var_es_t <- function(mu, s, nu, p) {   # r = mu + s z, con z t de Student estandarizada
  cc <- sqrt((nu - 2) / nu); q <- qt(p, nu); es_t <- dt(q, nu) / (1 - p) * (nu + q^2) / (nu - 1)
  c(VaR = -mu + s * cc * q, ES = -mu + s * cc * es_t)
}
var_es_hist <- function(L, p) { v <- unname(quantile(L, p, type = 7)); c(VaR = v, ES = mean(L[L >= v])) }
kupiec <- function(x, p) {             # cobertura incondicional; x = 1 si la pérdida superó al VaR
  n <- length(x); k <- sum(x); pi <- k / n; q <- 1 - p
  l0 <- (n - k) * log(1 - q) + k * log(q); l1 <- (n - k) * log(1 - pi) + if (k > 0) k * log(pi) else 0
  LR <- -2 * (l0 - l1); c(LR = LR, p = pchisq(LR, 1, lower.tail = FALSE))
}
christoffersen <- function(x) {        # independencia de las excepciones (cadena de Markov)
  a <- x[-length(x)]; b <- x[-1]
  n00 <- sum(a == 0 & b == 0); n01 <- sum(a == 0 & b == 1); n10 <- sum(a == 1 & b == 0); n11 <- sum(a == 1 & b == 1)
  p01 <- n01 / (n00 + n01); p11 <- if (n10 + n11 > 0) n11 / (n10 + n11) else 0; p1 <- (n01 + n11) / (n00 + n01 + n10 + n11)
  xl <- function(n, q) if (n > 0) n * log(q) else 0
  l0 <- xl(n00 + n10, 1 - p1) + xl(n01 + n11, p1)
  l1 <- xl(n00, 1 - p01) + xl(n01, p01) + xl(n10, 1 - p11) + xl(n11, p11)
  LR <- -2 * (l0 - l1); c(LR = LR, p = pchisq(LR, 1, lower.tail = FALSE), n00 = n00, n01 = n01, n10 = n10, n11 = n11)
}
gpd_nll <- function(par, y) {          # menos log-verosimilitud de la Pareto generalizada
  xi <- par[1]; beta <- par[2]
  if (beta <= 0) return(1e10)
  if (abs(xi) < 1e-8) return(length(y) * log(beta) + sum(y) / beta)
  w <- 1 + xi * y / beta
  if (any(w <= 0)) return(1e10)
  length(y) * log(beta) + (1 + 1 / xi) * sum(log(w))
}
gpd_fit <- function(y) {
  o <- optim(c(0.1, mean(y)), gpd_nll, y = y, method = "Nelder-Mead", control = list(reltol = 1e-14, maxit = 20000))
  p <- o$par; h <- 1e-5 * pmax(abs(p), 1); E <- diag(h, 2); H <- matrix(0, 2, 2); f <- function(q) gpd_nll(q, y)
  for (i in 1:2) for (j in 1:2)
    H[i, j] <- (f(p + E[, i] + E[, j]) - f(p + E[, i] - E[, j]) - f(p - E[, i] + E[, j]) + f(p - E[, i] - E[, j])) / (4 * h[i] * h[j])
  list(xi = p[1], beta = p[2], ee = sqrt(diag(solve(H))), ll = -o$value)
}
gpd_var_es <- function(u, xi, beta, n, nu, p) {
  v <- u + beta / xi * (((n / nu) * (1 - p))^(-xi) - 1); c(VaR = v, ES = v / (1 - xi) + (beta - xi * u) / (1 - xi))
}
