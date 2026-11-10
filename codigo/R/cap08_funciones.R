# Capítulo 8: funciones propias para estimar un GARCH(1,1) por máxima verosimilitud
# Parámetros: p = (mu, omega, alpha, beta) y, si dist = "t", nu como quinto elemento.
backcast <- function(e) {                 # varianza inicial: promedio ponderado de los primeros e^2
  tau <- min(75, length(e)); w <- 0.94^(0:(tau - 1)); w <- w / sum(w)
  sum(w * e[1:tau]^2)
}
garch_s2 <- function(p, r, bc) {          # recursión sigma2_t = omega + alpha e_{t-1}^2 + beta sigma2_{t-1}
  e <- r - p[1]; n <- length(e); s2 <- numeric(n)
  s2[1] <- p[2] + (p[3] + p[4]) * bc
  for (t in 2:n) s2[t] <- p[2] + p[3] * e[t - 1]^2 + p[4] * s2[t - 1]
  s2
}
garch_llt <- function(p, r, bc, dist = "normal") {   # log-verosimilitud de cada observación
  e <- r - p[1]; s2 <- garch_s2(p, r, bc)
  if (dist == "normal") return(-0.5 * (log(2 * pi) + log(s2) + e^2 / s2))
  nu <- p[5]                                          # t de Student estandarizada (varianza 1)
  lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(pi * (nu - 2)) - 0.5 * log(s2) -
    (nu + 1) / 2 * log(1 + e^2 / (s2 * (nu - 2)))
}
garch_fit <- function(r, dist = "normal") {
  bc <- backcast(r - mean(r))
  p0 <- c(mean(r), 0.05 * var(r), 0.08, 0.90); lo <- c(-Inf, 1e-8, 0, 0); hi <- c(Inf, Inf, 1, 1)
  if (dist == "t") { p0 <- c(p0, 8); lo <- c(lo, 2.05); hi <- c(hi, 200) }
  f <- function(p) -sum(garch_llt(p, r, bc, dist))
  o <- nlminb(p0, f, lower = lo, upper = hi,
              scale = 1 / pmax(abs(p0), 0.01), control = list(eval.max = 2000, iter.max = 1000, rel.tol = 1e-14))
  p <- o$par; k <- length(p); h <- 1e-5 * pmax(abs(p), 1)
  L <- function(q) sum(garch_llt(q, r, bc, dist)); E <- diag(h, k)
  H <- matrix(0, k, k)                                                # hessiana numérica
  for (i in 1:k) for (j in 1:k)
    H[i, j] <- (L(p + E[, i] + E[, j]) - L(p + E[, i] - E[, j]) - L(p - E[, i] + E[, j]) +
                L(p - E[, i] - E[, j])) / (4 * h[i] * h[j])
  G <- sapply(1:k, function(i) { d <- replace(numeric(k), i, h[i])   # scores de cada observación
    (garch_llt(p + d, r, bc, dist) - garch_llt(p - d, r, bc, dist)) / (2 * h[i]) })
  Hi <- solve(-H)
  nm <- c("mu", "omega", "alpha", "beta", "nu")[1:k]
  list(par = setNames(p, nm), ll = -o$objective, bc = bc, s2 = garch_s2(p, r, bc),
       ee = setNames(sqrt(diag(Hi)), nm), ee_rob = setNames(sqrt(diag(Hi %*% crossprod(G) %*% Hi)), nm))
}
