# Capítulo 12: funciones propias para volatilidad multivariada (EWMA, CCC y DCC en dos etapas)
source("codigo/R/cap08_funciones.R")
panel_diario <- function() {      # rendimientos logarítmicos diarios (%) del S&P 500, el Nasdaq y el Brent
  s <- read.csv("datos/internacional/sp500_diario_yahoo.csv")[, c("Date", "Adj.Close")]; names(s)[2] <- "sp"
  n <- read.csv("datos/internacional/nasdaq_diario_yahoo.csv")[, c("Date", "Adj.Close")]; names(n)[2] <- "nq"
  b <- read.csv("datos/internacional/brent_diario.csv"); names(b)[2] <- "br"
  d <- merge(merge(s, n, by = "Date"), b, by = "Date")
  list(fecha = d$Date[-1], R = 100 * diff(log(as.matrix(d[, c("sp", "nq", "br")]))))
}
ewma_cov <- function(R, lam = 0.94) {   # H_t = lam H_{t-1} + (1 - lam) e_{t-1} e_{t-1}'
  E <- sweep(R, 2, colMeans(R)); T <- nrow(E); K <- ncol(E); H <- array(0, c(T, K, K)); H[1, , ] <- cov(E[1:30, ])
  for (t in 2:T) H[t, , ] <- lam * H[t - 1, , ] + (1 - lam) * tcrossprod(E[t - 1, ])
  H
}
corr_de_cov <- function(Q) { d <- 1 / sqrt(diag(Q)); Q * outer(d, d) }
etapa1 <- function(R) {           # GARCH(1,1) normal para cada serie (Capítulo 8)
  ms <- lapply(1:ncol(R), function(i) garch_fit(R[, i], "normal"))
  list(Z = sapply(1:ncol(R), function(i) (R[, i] - ms[[i]]$par["mu"]) / sqrt(ms[[i]]$s2)),
       S2 = sapply(ms, function(m) m$s2), modelos = ms)
}
dcc_filtro <- function(p, Z, Qb = crossprod(Z) / nrow(Z)) {   # Q_t = (1-a-b) Qbar + a z z' + b Q_{t-1}
  a <- p[1]; b <- p[2]; T <- nrow(Z); K <- ncol(Z); Q <- Qb; Rt <- array(0, c(T, K, K))
  for (t in 1:T) {
    if (t > 1) Q <- (1 - a - b) * Qb + a * tcrossprod(Z[t - 1, ]) + b * Q
    Rt[t, , ] <- corr_de_cov(Q)
  }
  Rt
}
ll_corr <- function(Rt, Z) {      # -1/2 [ ln|R_t| + z_t' R_t^{-1} z_t - z_t' z_t ]
  sapply(1:nrow(Z), function(t) { R <- Rt[t, , ]; z <- Z[t, ]
    -0.5 * (determinant(R)$modulus + sum(z * solve(R, z)) - sum(z^2)) })
}
dcc_fit <- function(Z) {
  f <- function(p) if (p[1] + p[2] >= 0.9999) 1e10 else -sum(ll_corr(dcc_filtro(p, Z), Z))
  o <- nlminb(c(0.03, 0.95), f, lower = c(1e-6, 0), upper = c(0.5, 0.9999), control = list(rel.tol = 1e-14))
  p <- o$par; h <- 1e-5 * pmax(abs(p), 1); E <- diag(h, 2); H <- matrix(0, 2, 2)
  L <- function(q) sum(ll_corr(dcc_filtro(q, Z), Z))
  for (i in 1:2) for (j in 1:2)
    H[i, j] <- (L(p + E[, i] + E[, j]) - L(p + E[, i] - E[, j]) - L(p - E[, i] + E[, j]) + L(p - E[, i] - E[, j])) / (4 * h[i] * h[j])
  Rc <- cor(Z); Rcc <- array(rep(Rc, each = nrow(Z)), c(nrow(Z), ncol(Z), ncol(Z)))
  list(a = p[1], b = p[2], ee = sqrt(diag(solve(-H))), ll = L(p), ll_ccc = sum(ll_corr(Rcc, Z)), R = dcc_filtro(p, Z), Rc = Rc)
}
