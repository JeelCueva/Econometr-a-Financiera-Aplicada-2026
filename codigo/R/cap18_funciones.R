# Capítulo 18: funciones propias para cópulas
indices <- function() {
  a <- read.csv("datos/internacional/sp500_diario_yahoo.csv"); b <- read.csv("datos/internacional/nasdaq_diario_yahoo.csv")
  list(fecha = a$Date[-1], x = 100 * diff(log(a$Adj.Close)), y = 100 * diff(log(b$Adj.Close)))
}
pseudo <- function(x) rank(x) / (length(x) + 1)
ewma_estandarizar <- function(x, lam = 0.94, n0 = 30) {
  h <- numeric(length(x)); h[1] <- mean(x[1:n0]^2)
  for (i in 2:length(x)) h[i] <- lam * h[i - 1] + (1 - lam) * x[i - 1]^2
  x / sqrt(h)
}
ll_gauss <- function(r, u, v) { x <- qnorm(u); y <- qnorm(v); sum(-0.5 * log(1 - r^2) - (r^2 * (x^2 + y^2) - 2 * r * x * y) / (2 * (1 - r^2))) }
ll_t <- function(r, nu, u, v) {
  x <- qt(u, nu); y <- qt(v, nu); q <- (x^2 + y^2 - 2 * r * x * y) / (1 - r^2)
  cc <- lgamma((nu + 2) / 2) + lgamma(nu / 2) - 2 * lgamma((nu + 1) / 2) - 0.5 * log(1 - r^2)
  sum(cc - (nu + 2) / 2 * log(1 + q / nu) + (nu + 1) / 2 * (log(1 + x^2 / nu) + log(1 + y^2 / nu)))
}
ll_clayton <- function(th, u, v) sum(log(1 + th) - (th + 1) * (log(u) + log(v)) - (2 + 1 / th) * log(u^-th + v^-th - 1))
ll_gumbel <- function(th, u, v) {
  x <- -log(u); y <- -log(v); s <- x^th + y^th; A <- s^(1 / th)
  sum(-A + x + y + (th - 1) * (log(x) + log(y)) + (1 / th - 2) * log(s) + log(A + th - 1))
}
dorada <- function(f, a, b, tol = 1e-9) {
  g <- (sqrt(5) - 1) / 2; c <- b - g * (b - a); d <- a + g * (b - a); fc <- f(c); fd <- f(d)
  while (b - a > tol) {
    if (fc > fd) { b <- d; d <- c; fd <- fc; c <- b - g * (b - a); fc <- f(c) }
    else { a <- c; c <- d; fc <- fd; d <- a + g * (b - a); fd <- f(d) }
  }
  x <- (a + b) / 2; c(x, f(x))
}
ajustar <- function(u, v) {
  out <- list(); out$gaussiana <- dorada(function(r) ll_gauss(r, u, v), -0.99, 0.99)
  prof <- function(nu) dorada(function(r) ll_t(r, nu, u, v), -0.99, 0.99)[2]
  nus <- 2:30; best <- nus[which.max(sapply(nus, prof))]
  nu <- dorada(prof, max(2, best - 1), best + 1, tol = 1e-6)[1]
  rr <- dorada(function(r) ll_t(r, nu, u, v), -0.99, 0.99); out$t <- c(rr[1], nu, rr[2])
  out$clayton <- dorada(function(th) ll_clayton(th, u, v), 0.01, 20)
  out$gumbel <- dorada(function(th) ll_gumbel(th, u, v), 1.0001, 20)
  out
}
tau_lambda <- function(nombre, p) {
  if (nombre == "gaussiana") return(c(2 / pi * asin(p), 0, 0))
  if (nombre == "t") { l <- 2 * pt(-sqrt((p[2] + 1) * (1 - p[1]) / (1 + p[1])), p[2] + 1); return(c(2 / pi * asin(p[1]), l, l)) }
  if (nombre == "clayton") return(c(p / (p + 2), 2^(-1 / p), 0))
  if (nombre == "gumbel") return(c(1 - 1 / p, 0, 2 - 2^(1 / p)))
}
colas_empiricas <- function(u, v, q) { n <- length(u); c(sum(u <= q & v <= q) / (n * q), sum(u > 1 - q & v > 1 - q) / (n * q)) }
lcg_nuevo <- function(semilla) { e <- new.env(); e$x <- semilla; e }
lcg_u <- function(g, n = 1) { z <- numeric(n); for (i in 1:n) { g$x <- (16807 * g$x) %% 2147483647; z[i] <- g$x / 2147483647 }; z }
phi2_diag <- function(a, r, m = 20001) {
  x <- seq(-10, a, length.out = m); f <- dnorm(x) * pnorm((a - r * x) / sqrt(1 - r^2)); h <- x[2] - x[1]
  h / 3 * (f[1] + f[m] + 4 * sum(f[seq(2, m - 1, by = 2)]) + 2 * sum(f[seq(3, m - 2, by = 2)]))
}
