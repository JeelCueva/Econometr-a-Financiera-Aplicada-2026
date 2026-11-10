# Capítulo 10: funciones propias para estimar y analizar un VAR(p) por MCO
rezagos <- function(Y, p, inicio = p) {   # regresores [1, y_{t-1}', ..., y_{t-p}'] y regresando y_t
  Y <- as.matrix(Y); T <- nrow(Y); s <- inicio
  X <- cbind(1, do.call(cbind, lapply(1:p, function(j) Y[(s - j + 1):(T - j), , drop = FALSE])))
  list(y = Y[(s + 1):T, , drop = FALSE], X = X)
}
var_mco <- function(Y, p, inicio = p) {
  d <- rezagos(Y, p, inicio); y <- d$y; X <- d$X; n <- nrow(y); K <- ncol(y)
  B <- solve(crossprod(X), crossprod(X, y))               # (1 + Kp) x K: una columna por ecuación
  U <- y - X %*% B
  list(B = B, U = U, X = X, y = y, p = p, K = K, n = n,
       S = crossprod(U) / (n - ncol(X)),                  # Sigma con corrección de grados de libertad
       S_mv = crossprod(U) / n,                           # Sigma de máxima verosimilitud
       A = lapply(1:p, function(j) t(B[1 + K * (j - 1) + 1:K, , drop = FALSE])))   # A_j es K x K
}
seleccion <- function(Y, pmax) {          # criterios de información con la misma muestra para todos los p
  t(sapply(0:pmax, function(p) {
    if (p == 0) { y <- as.matrix(Y)[(pmax + 1):nrow(Y), ]; U <- sweep(y, 2, colMeans(y)); n <- nrow(y)
                  K <- ncol(y); S <- crossprod(U) / n }
    else { m <- var_mco(Y, p, inicio = pmax); n <- m$n; K <- m$K; S <- m$S_mv }
    ld <- log(det(S)); k <- p * K^2 + K
    c(p = p, AIC = ld + 2 * k / n, BIC = ld + log(n) * k / n, HQ = ld + 2 * log(log(n)) * k / n)
  }))
}
companera <- function(A) {
  K <- nrow(A[[1]]); p <- length(A); F <- matrix(0, K * p, K * p)
  F[1:K, ] <- do.call(cbind, A)
  if (p > 1) F[(K + 1):(K * p), 1:(K * (p - 1))] <- diag(K * (p - 1))
  F
}
irf <- function(A, h, P = NULL) {         # lista Psi_0..Psi_h, o Theta_i = Psi_i P si se da P
  K <- nrow(A[[1]]); Psi <- list(diag(K))
  for (i in 1:h) {
    s <- matrix(0, K, K)
    for (j in 1:min(i, length(A))) s <- s + A[[j]] %*% Psi[[i - j + 1]]
    Psi[[i + 1]] <- s
  }
  if (is.null(P)) Psi else lapply(Psi, function(M) M %*% P)
}
fevd <- function(Theta) {                 # proporciones [h, variable k, choque j]
  H <- length(Theta); K <- nrow(Theta[[1]]); out <- array(0, c(H, K, K)); acum <- matrix(0, K, K)
  for (i in 1:H) { acum <- acum + Theta[[i]]^2; out[i, , ] <- acum / rowSums(acum) }
  out
}
granger <- function(Y, p, efecto, causas) {   # F de una ecuación y Wald robusto (HC0); índices desde 1
  d <- rezagos(Y, p); y <- d$y[, efecto]; X <- d$X; n <- length(y); q <- ncol(X); K <- ncol(Y)
  quitar <- unlist(lapply(1:p, function(j) 1 + K * (j - 1) + causas))
  b <- solve(crossprod(X), crossprod(X, y)); e <- y - X %*% b; ssr_u <- sum(e^2)
  Xr <- X[, -quitar, drop = FALSE]; er <- y - Xr %*% solve(crossprod(Xr), crossprod(Xr, y))
  r <- length(quitar); F <- ((sum(er^2) - ssr_u) / r) / (ssr_u / (n - q))
  XtXi <- solve(crossprod(X)); V <- XtXi %*% crossprod(X * as.vector(e)) %*% XtXi     # HC0
  W <- as.numeric(t(b[quitar]) %*% solve(V[quitar, quitar], b[quitar]))
  c(F = F, gl1 = r, gl2 = n - q, pF = pf(F, r, n - q, lower.tail = FALSE),
    W = W, pW = pchisq(W, r, lower.tail = FALSE))
}
bootstrap_irf <- function(Y, p, h, B = 500, semilla = 1) {   # bandas al 90 % (percentiles 5 y 95)
  Y <- as.matrix(Y); m <- var_mco(Y, p); set.seed(semilla); n <- m$n; K <- m$K
  sal <- array(0, c(B, h + 1, K, K))
  for (b in 1:B) {
    U <- m$U[sample.int(n, n, replace = TRUE), , drop = FALSE]; Yb <- Y
    for (t in (p + 1):nrow(Y)) {
      v <- m$B[1, ]
      for (j in 1:p) v <- v + m$A[[j]] %*% Yb[t - j, ]
      Yb[t, ] <- v + U[t - p, ]
    }
    mb <- var_mco(Yb, p); th <- irf(mb$A, h, t(chol(mb$S)))
    for (i in 1:(h + 1)) sal[b, i, , ] <- th[[i]]
  }
  list(inf = apply(sal, 2:4, quantile, 0.05, type = 7), sup = apply(sal, 2:4, quantile, 0.95, type = 7))
}
