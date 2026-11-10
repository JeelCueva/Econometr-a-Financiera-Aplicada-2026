# Capítulo 9: funciones propias para GARCH, GJR-GARCH, EGARCH y GARCH en media (máxima verosimilitud)
# Vector de parámetros: mu, [lambda si media = "m"], omega, alpha, [gamma si GJR o EGARCH], beta, [nu si dist = "t"]
RAIZ_2_PI <- sqrt(2 / pi)                                 # E|z| para z normal estándar
backcast <- function(e) {                                 # igual que en el Capítulo 8
  tau <- min(75, length(e)); w <- 0.94^(0:(tau - 1)); w <- w / sum(w)
  sum(w * e[1:tau]^2)
}
desarmar <- function(p, modelo, media) {
  k <- if (media == "m") 3 else 2
  lam <- if (media == "m") p[2] else 0
  if (modelo == "garch") { v <- c(p[k], p[k + 1], 0, p[k + 2]); k <- k + 3 }
  else                   { v <- p[k:(k + 3)];               k <- k + 4 }
  list(mu = p[1], lam = lam, om = v[1], a = v[2], g = v[3], b = v[4], nu = if (length(p) >= k) p[k] else NA)
}
filtro <- function(p, r, bc, modelo = "garch", media = "c") {   # innovaciones e_t y varianzas sigma2_t
  q <- desarmar(p, modelo, media); n <- length(r); e <- s2 <- numeric(n)
  for (t in 1:n) {
    if (modelo == "egarch") {
      if (t == 1) ls <- q$om + q$b * log(bc)
      else { z <- e[t - 1] / sqrt(s2[t - 1])
             ls <- q$om + q$a * (abs(z) - RAIZ_2_PI) + q$g * z + q$b * ls }
      s2[t] <- exp(ls)
    } else {                                              # GARCH (g = 0) y GJR-GARCH
      s2[t] <- if (t == 1) q$om + (q$a + q$g / 2 + q$b) * bc
               else q$om + (q$a + q$g * (e[t - 1] < 0)) * e[t - 1]^2 + q$b * s2[t - 1]
    }
    e[t] <- r[t] - q$mu - q$lam * s2[t]
  }
  list(e = e, s2 = s2)
}
llt <- function(p, r, bc, modelo = "garch", media = "c", dist = "normal") {
  f <- filtro(p, r, bc, modelo, media); e <- f$e; s2 <- f$s2
  if (dist == "normal") return(-0.5 * (log(2 * pi) + log(s2) + e^2 / s2))
  nu <- p[length(p)]
  lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(pi * (nu - 2)) - 0.5 * log(s2) -
    (nu + 1) / 2 * log(1 + e^2 / (s2 * (nu - 2)))
}
ajustar <- function(r, modelo = "garch", media = "c", dist = "normal") {
  bc <- backcast(r - mean(r)); v <- var(r)
  p0 <- mean(r); lo <- -Inf; hi <- Inf
  if (media == "m") { p0 <- c(p0, 0); lo <- c(lo, -0.5); hi <- c(hi, 0.5) }
  if (modelo == "garch")  { p0 <- c(p0, 0.05 * v, 0.08, 0.90);         lo <- c(lo, 1e-8, 0, 0);     hi <- c(hi, Inf, 1, 1) }
  if (modelo == "gjr")    { p0 <- c(p0, 0.05 * v, 0.03, 0.08, 0.88);   lo <- c(lo, 1e-8, 0, -1, 0); hi <- c(hi, Inf, 1, 1, 1) }
  if (modelo == "egarch") { p0 <- c(p0, 0.03 * log(v), 0.10, -0.08, 0.97); lo <- c(lo, -Inf, 0, -1, 0); hi <- c(hi, Inf, 2, 1, 0.9999) }
  if (dist == "t") { p0 <- c(p0, 8); lo <- c(lo, 2.05); hi <- c(hi, 200) }
  f <- function(p) {                                      # penaliza parámetros que hacen explotar la recursión
    v <- -sum(llt(p, r, bc, modelo, media, dist)); if (is.finite(v)) v else 1e12 }
  o <- nlminb(p0, f, lower = lo, upper = hi, scale = 1 / pmax(abs(p0), 0.01),
              control = list(eval.max = 3000, iter.max = 2000, rel.tol = 1e-14))
  p <- o$par; k <- length(p)
  L <- function(q) sum(llt(q, r, bc, modelo, media, dist))
  hess <- function(p) {
    h <- 1e-5 * pmax(abs(p), 1); E <- diag(h, k); H <- matrix(0, k, k)
    g <- sapply(1:k, function(i) (L(p + E[, i]) - L(p - E[, i])) / (2 * h[i]))
    for (i in 1:k) for (j in 1:k)
      H[i, j] <- (L(p + E[, i] + E[, j]) - L(p + E[, i] - E[, j]) - L(p - E[, i] + E[, j]) +
                  L(p - E[, i] - E[, j])) / (4 * h[i] * h[j])
    list(g = g, H = H)
  }
  for (it in 1:20) {                                      # pulido: Newton con reducción del paso
    d <- hess(p); paso <- solve(d$H, d$g); mejora <- FALSE
    for (m in 1:12) {
      q <- p - paso
      if (all(q >= lo & q <= hi) && is.finite(L(q)) && L(q) > L(p)) { mejora <- TRUE; break }
      paso <- paso / 2
    }
    if (!mejora) break
    dif <- L(q) - L(p); p <- q
    if (dif < 1e-10) break
  }
  h <- 1e-5 * pmax(abs(p), 1); E <- diag(h, k)
  G <- sapply(1:k, function(i) (llt(p + E[, i], r, bc, modelo, media, dist) -
                                llt(p - E[, i], r, bc, modelo, media, dist)) / (2 * h[i]))
  H <- matrix(0, k, k)
  for (i in 1:k) for (j in 1:k)
    H[i, j] <- (L(p + E[, i] + E[, j]) - L(p + E[, i] - E[, j]) - L(p - E[, i] + E[, j]) +
                L(p - E[, i] - E[, j])) / (4 * h[i] * h[j])
  Hi <- solve(-H)
  nm <- c("mu", if (media == "m") "lambda", "omega", "alpha", if (modelo != "garch") "gamma", "beta",
          if (dist == "t") "nu")
  fl <- filtro(p, r, bc, modelo, media)
  list(p = setNames(p, nm), ll = L(p), bc = bc, e = fl$e, s2 = fl$s2, k = k,
       ee_rob = setNames(sqrt(diag(Hi %*% crossprod(G) %*% Hi)), nm), ee = setNames(sqrt(diag(Hi)), nm))
}
