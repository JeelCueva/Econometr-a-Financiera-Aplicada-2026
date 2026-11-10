# Capítulo 11: funciones propias para cointegración (Engle-Granger, Johansen y VECM)
source("codigo/R/cap05_funciones.R")
crudo_mensual <- function() {      # promedios mensuales de los precios diarios del Brent y del WTI
  b <- read.csv("datos/internacional/brent_diario.csv"); w <- read.csv("datos/internacional/wti_diario.csv")
  d <- merge(b, w, by = "Date", suffixes = c("_b", "_w")); d$mes <- substr(d$Date, 1, 7)
  m <- aggregate(cbind(Price_b, Price_w) ~ mes, data = d, FUN = mean)
  m[m$mes <= "2026-10", ]                                 # se excluye el mes incompleto de noviembre
}
vc_engle_granger <- function(T) {  # MacKinnon (2010), 2 variables con constante
  b <- rbind("1%" = c(-3.89644, -10.9519, -22.527), "5%" = c(-3.33613, -6.1101, -6.823),
             "10%" = c(-3.04445, -4.2412, -2.720))
  b[, 1] + b[, 2] / T + b[, 3] / T^2
}
engle_granger <- function(y, x) {  # paso 1: MCO de y sobre (1, x); paso 2: ADF sin constante sobre los residuos
  m <- lm(y ~ x); u <- as.numeric(residuals(m)); r <- adf(u, det = "n")
  list(a = coef(m)[1], b = coef(m)[2], u = u, t = r$t, p = r$p, gamma = r$gamma, vc = vc_engle_granger(length(y)))
}
resid_mat <- function(Y, Z) Y - Z %*% solve(crossprod(Z), crossprod(Z, Y))
johansen <- function(Y, k) {       # constante sin restringir y k rezagos de las diferencias
  Y <- as.matrix(Y); dY <- diff(Y); n <- nrow(dY) - k
  Z <- cbind(matrix(1, n, 1), if (k > 0) do.call(cbind, lapply(1:k, function(j) dY[(k - j + 1):(nrow(dY) - j), , drop = FALSE])))
  dYt <- dY[(k + 1):nrow(dY), , drop = FALSE]; Ylag <- Y[(k + 1):(nrow(Y) - 1), , drop = FALSE]
  R0 <- resid_mat(dYt, Z); R1 <- resid_mat(Ylag, Z)        # Delta y_t y y_{t-1}, depurados de Z
  S00 <- crossprod(R0) / n; S01 <- crossprod(R0, R1) / n; S11 <- crossprod(R1) / n
  e <- eigen(solve(S11, t(S01) %*% solve(S00, S01)))
  lam <- Re(e$values); V <- Re(e$vectors); o <- order(lam, decreasing = TRUE); lam <- lam[o]; V <- V[, o]
  V <- sweep(V, 2, sqrt(diag(t(V) %*% S11 %*% V)), "/")  # normalización beta' S11 beta = I
  K <- ncol(Y)
  list(lam = lam, V = V, n = n, S00 = S00, S01 = S01, S11 = S11, Z = Z, dY = dYt, Ylag = Ylag,
       traza = sapply(0:(K - 1), function(r) -n * sum(log(1 - lam[(r + 1):K]))),
       maxv = sapply(0:(K - 1), function(r) -n * log(1 - lam[r + 1])))
}
# Valores críticos al 90, 95 y 99 % (MacKinnon, Haug y Michelis, 1999); fila = K - r
vc_traza <- rbind(c(2.7055, 3.8415, 6.6349), c(13.4294, 15.4943, 19.9349), c(27.0669, 29.7961, 35.4628))
vc_maxv  <- rbind(c(2.7055, 3.8415, 6.6349), c(12.2971, 14.2639, 18.5200), c(18.8928, 21.1314, 25.8650))
vecm <- function(Y, k, r = 1) {    # beta de Johansen normalizada; alpha, Gamma y constante por MCO
  J <- johansen(Y, k); beta <- J$V[, 1:r, drop = FALSE] / J$V[1, 1:r]
  ect <- J$Ylag %*% beta; X <- cbind(ect, J$Z)
  B <- solve(crossprod(X), crossprod(X, J$dY)); U <- J$dY - X %*% B; n <- nrow(X); q <- ncol(X)
  S <- crossprod(U) / (n - q); ee <- sqrt(outer(diag(solve(crossprod(X))), diag(S)))
  list(beta = beta, alpha = t(B[1:r, , drop = FALSE]), ee_alpha = t(ee[1:r, , drop = FALSE]),
       B = B, ee = ee, U = U, S = S, J = J, ect = ect)
}
lr_beta <- function(Y, k, H) {     # razón de verosimilitudes de beta = H phi (r = 1)
  J <- johansen(Y, k); S11h <- t(H) %*% J$S11 %*% H; S01h <- J$S01 %*% H
  lh <- max(Re(eigen(solve(S11h, t(S01h) %*% solve(J$S00, S01h)))$values))
  c(LR = J$n * (log(1 - lh) - log(1 - J$lam[1])), lam = J$lam[1], lam_h = lh)
}
