# Capítulo 14: funciones propias para modelos de factores (regresiones de series de tiempo, GRS, Fama-MacBeth)
VALOR <- c("S1V1", "S1V3", "S1V5", "S3V1", "S3V3", "S3V5", "S5V1", "S5V3", "S5V5")
MOMENTUM <- c("S1M1", "S1M3", "S1M5", "S3M1", "S3M3", "S3M5", "S5M1", "S5M3", "S5M5")
INDUSTRIAS <- c("NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other")
datos <- function() {   # excesos de rendimiento mensuales (%) y factores en %, 1949-01 a 2017-03
  f <- read.csv("datos/internacional/french_mensual_1949_2017.csv")
  list(fecha = f$dates, R = 100 * (as.matrix(f[, c(VALOR, MOMENTUM, INDUSTRIAS)]) - f$RF),
       F = 100 * as.matrix(f[, c("MktRF", "SMB", "HML", "Mom")]))
}
mcov <- function(X) { X <- as.matrix(X); crossprod(sweep(X, 2, colMeans(X))) / nrow(X) }   # covarianza MV
regs_ts <- function(R, F) {   # MCO de cada cartera sobre una constante y los factores
  R <- as.matrix(R); F <- as.matrix(F); T <- nrow(F); L <- ncol(F); X <- cbind(1, F)
  B <- solve(crossprod(X), crossprod(X, R)); E <- R - X %*% B; S <- crossprod(E) / (T - L - 1)
  ee <- sqrt(outer(diag(solve(crossprod(X))), diag(S)))
  list(alfa = B[1, ], beta = t(B[-1, , drop = FALSE]), t_alfa = B[1, ] / ee[1, ], E = E, S = S,
       r2 = 1 - colSums(E^2) / colSums(sweep(R, 2, colMeans(R))^2))
}
grs <- function(R, F) {       # Gibbons, Ross y Shanken (1989)
  R <- as.matrix(R); F <- as.matrix(F); T <- nrow(R); N <- ncol(R); L <- ncol(F)
  m <- regs_ts(R, F); mu <- colMeans(F); Om <- mcov(F)
  a2 <- sum(m$alfa * solve(m$S, m$alfa)); sf2 <- sum(mu * solve(Om, mu))
  Fs <- (T - N - L) / N * T / (T - L - 1) * a2 / (1 + sf2)
  list(F = Fs, p = pf(Fs, N, T - N - L, lower.tail = FALSE), gl = c(N, T - N - L), sr2_f = sf2,
       alfa_abs = mean(abs(m$alfa)), a2_mv = a2 * T / (T - L - 1))
}
sr2 <- function(X) { mu <- colMeans(as.matrix(X)); sum(mu * solve(mcov(X), mu)) }   # Sharpe máximo al cuadrado
fama_macbeth <- function(R, F) {
  R <- as.matrix(R); F <- as.matrix(F); T <- nrow(R); N <- ncol(R); L <- ncol(F)
  m <- regs_ts(R, F); X <- cbind(1, m$beta)
  G <- t(solve(crossprod(X), crossprod(X, t(R))))                # T x (1 + L)
  lam <- colMeans(G); ee_fm <- apply(G, 2, sd) / sqrt(T)
  Om <- mcov(F); c <- sum(lam[-1] * solve(Om, lam[-1]))
  A <- solve(crossprod(X), t(X)); Os <- matrix(0, L + 1, L + 1); Os[-1, -1] <- Om
  V_sh <- (A %*% m$S %*% t(A) * (1 + c) + Os) / T                # corrección de Shanken (1992)
  media <- colMeans(R); ajuste <- X %*% lam
  list(lam = lam, ee_fm = ee_fm, ee_sh = sqrt(diag(V_sh)), r2_ct = 1 - sum((media - ajuste)^2) / sum((media - mean(media))^2),
       beta = m$beta, c = c)
}
