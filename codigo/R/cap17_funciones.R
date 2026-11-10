# Capítulo 17: funciones propias para modelos con cambios de régimen de Markov
mercado <- function() {
  d <- read.csv("datos/internacional/french_mercado_mensual_1926_2018.csv", check.names = FALSE)
  list(fecha = d$dates, y = d[["Mkt-RF"]])
}
normal <- function(y, mu, s) exp(-0.5 * ((y - mu) / s)^2) / (s * sqrt(2 * pi))
filtro_hamilton <- function(D, P, rho) {
  T <- nrow(D); K <- ncol(D); xf <- xp <- matrix(0, T, K); ll <- 0
  for (t in 1:T) {
    xp[t, ] <- if (t == 1) rho else as.vector(xf[t - 1, ] %*% P)
    num <- xp[t, ] * D[t, ]; f <- sum(num); xf[t, ] <- num / f; ll <- ll + log(f)
  }
  list(xf = xf, xp = xp, ll = ll)
}
suavizado_kim <- function(xf, xp, P) {
  T <- nrow(xf); K <- ncol(xf); xs <- matrix(0, T, K); xs[T, ] <- xf[T, ]; N <- matrix(0, K, K)
  for (t in (T - 1):1) {
    r <- xs[t + 1, ] / xp[t + 1, ]
    xs[t, ] <- xf[t, ] * as.vector(P %*% r)
    N <- N + (xf[t, ] * P) * matrix(r, K, K, byrow = TRUE)
  }
  list(xs = xs, N = N)
}
em_media_varianza <- function(y, K, mu0, s0, P0, media_comun = FALSE, tol = 1e-10, maxit = 10000) {
  mu <- mu0; s <- s0; P <- P0; rho <- rep(1 / K, K); ll0 <- -Inf
  for (it in 0:(maxit - 1)) {
    D <- sapply(1:K, function(j) normal(y, mu[j], s[j]))
    h <- filtro_hamilton(D, P, rho); k <- suavizado_kim(h$xf, h$xp, P); xs <- k$xs
    if (h$ll - ll0 < tol) break
    ll0 <- h$ll; w <- colSums(xs)
    mu <- if (media_comun) rep(sum(xs * y) / sum(w), K) else colSums(xs * y) / w
    s <- sqrt(colSums(xs * outer(y, mu, "-")^2) / w)
    P <- k$N / rowSums(k$N); rho <- xs[1, ]
  }
  list(mu = mu, s = s, P = P, rho = rho, ll = h$ll, xf = h$xf, xp = h$xp, xs = xs, it = it)
}
em_msiar <- function(y, K, c0, phi0, s0, P0, tol = 1e-10, maxit = 10000) {
  z <- y[-1]; x <- y[-length(y)]; T <- length(z)
  cc <- c0; phi <- phi0; s <- s0; P <- P0; rho <- rep(1 / K, K); ll0 <- -Inf
  for (it in 0:(maxit - 1)) {
    D <- sapply(1:K, function(j) normal(z, cc[j] + phi * x, s))
    h <- filtro_hamilton(D, P, rho); k <- suavizado_kim(h$xf, h$xp, P); xs <- k$xs
    if (h$ll - ll0 < tol) break
    ll0 <- h$ll; w <- colSums(xs)
    A <- matrix(0, K + 1, K + 1); b <- numeric(K + 1)
    A[1:K, 1:K] <- diag(w, K); A[1:K, K + 1] <- A[K + 1, 1:K] <- colSums(xs * x); A[K + 1, K + 1] <- sum(x^2)
    b[1:K] <- colSums(xs * z); b[K + 1] <- sum(x * z)
    th <- solve(A, b); cc <- th[1:K]; phi <- th[K + 1]
    s <- sqrt(sum(xs * (z - outer(phi * x, cc, "+"))^2) / T)
    P <- k$N / rowSums(k$N); rho <- xs[1, ]
  }
  list(c = cc, phi = phi, s = s, P = P, rho = rho, ll = h$ll, xf = h$xf, xp = h$xp, xs = xs, it = it)
}
ergodicas <- function(P) { K <- nrow(P); A <- rbind(t(P) - diag(K), 1); as.vector(qr.solve(A, c(rep(0, K), 1))) }
ll_mv <- function(th, y, rho) {
  P <- matrix(c(th[5], 1 - th[6], 1 - th[5], th[6]), 2, 2)
  filtro_hamilton(cbind(normal(y, th[1], th[3]), normal(y, th[2], th[4])), P, rho)$ll
}
hessiano <- function(f, th, h = 1e-4) {
  k <- length(th); H <- matrix(0, k, k); e <- diag(k) * h
  for (i in 1:k) for (j in 1:k)
    H[i, j] <- (f(th + e[i, ] + e[j, ]) - f(th + e[i, ] - e[j, ]) - f(th - e[i, ] + e[j, ]) + f(th - e[i, ] - e[j, ])) / (4 * h * h)
  H
}
recesion_mensual <- function(fechas) {
  r <- read.csv("datos/internacional/recesiones_nber.csv"); d <- integer(length(fechas))
  for (i in seq_len(nrow(r))) d[fechas > r$pico[i] & fechas <= r$valle[i]] <- 1L
  d
}
produccion <- function() {
  d <- read.csv("datos/internacional/produccion_industrial_eeuu_trimestral.csv"); list(trim = d$trimestre, y = d$produccion)
}
recesion_trimestral <- function(trim) {
  meses <- as.vector(sapply(trim, function(q) sprintf("%s-%02d", substr(q, 1, 4), 3 * (as.integer(substr(q, 6, 6)) - 1) + 1:3)))
  as.integer(colSums(matrix(recesion_mensual(meses), 3)) >= 2)
}
cuantil_mezcla <- function(a, w, mu, s, lo = -100, hi = 100, it = 200) {
  for (i in 1:it) { q <- 0.5 * (lo + hi); if (sum(w * pnorm((q - mu) / s)) < a) lo <- q else hi <- q }
  0.5 * (lo + hi)
}
kupiec_christoffersen <- function(h, a) {
  xl <- function(n, p) if (n > 0) n * log(p) else 0
  n <- length(h); x <- sum(h); p <- x / n
  uc <- -2 * (xl(n - x, 1 - a) + xl(x, a) - xl(n - x, 1 - p) - xl(x, p))
  a0 <- h[-n]; a1 <- h[-1]
  n00 <- sum(a0 == 0 & a1 == 0); n01 <- sum(a0 == 0 & a1 == 1); n10 <- sum(a0 == 1 & a1 == 0); n11 <- sum(a0 == 1 & a1 == 1)
  p01 <- n01 / (n00 + n01); p11 <- if (n10 + n11 > 0) n11 / (n10 + n11) else 0; p1 <- (n01 + n11) / (n00 + n01 + n10 + n11)
  ind <- -2 * (xl(n00 + n10, 1 - p1) + xl(n01 + n11, p1) - xl(n00, 1 - p01) - xl(n01, p01) - xl(n10, 1 - p11) - xl(n11, p11))
  c(uc, ind, n11)
}
