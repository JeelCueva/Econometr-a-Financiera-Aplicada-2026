# Capítulo 19: funciones propias para datos de panel
grunfeld <- function() { d <- read.csv("datos/internacional/grunfeld_inversion.csv")
  list(g = d$empresa, t = d$anio, y = d$inversion, X = as.matrix(d[, c("valor", "capital")])) }
mco <- function(y, X) { XtXi <- solve(crossprod(X)); b <- as.vector(XtXi %*% crossprod(X, y)); list(b = b, e = as.vector(y - X %*% b), XtXi = XtXi) }
media_grupo <- function(z, g) if (is.null(dim(z))) ave(z, g) else apply(z, 2, function(c) ave(c, g))
v_clasica <- function(X, e, XtXi, gl) sum(e^2) / gl * XtXi
v_white <- function(X, e, XtXi, k) { n <- length(e); n / (n - k) * XtXi %*% crossprod(X * e) %*% XtXi }
v_cluster <- function(X, e, XtXi, g, k) {
  n <- length(e); G <- length(unique(g)); S <- matrix(0, ncol(X), ncol(X))
  for (c in unique(g)) { s <- crossprod(X[g == c, , drop = FALSE], e[g == c]); S <- S + s %*% t(s) }
  G / (G - 1) * (n - 1) / (n - k) * XtXi %*% S %*% XtXi
}
v_dos_vias <- function(X, e, XtXi, g1, g2, k) v_cluster(X, e, XtXi, g1, k) + v_cluster(X, e, XtXi, g2, k) - v_white(X, e, XtXi, k)
v_driscoll_kraay <- function(X, e, XtXi, t, L) {
  ts <- sort(unique(t)); H <- t(sapply(ts, function(s) crossprod(X[t == s, , drop = FALSE], e[t == s])))
  if (ncol(X) == 1) H <- matrix(H, ncol = 1)
  S <- crossprod(H); Tn <- nrow(H)
  for (j in 1:L) { G <- t(H[(j + 1):Tn, , drop = FALSE]) %*% H[1:(Tn - j), , drop = FALSE]; S <- S + (1 - j / (L + 1)) * (G + t(G)) }
  XtXi %*% S %*% XtXi
}
efectos_aleatorios <- function(y, X, g) {
  N <- length(unique(g)); n <- nrow(X); K <- ncol(X); T <- n / N
  w <- mco(y - media_grupo(y, g), X - media_grupo(X, g)); s2e <- sum(w$e^2) / (n - N - K)
  idx <- !duplicated(g); yb <- media_grupo(y, g)[idx]; Xb <- media_grupo(X, g)[idx, , drop = FALSE]
  bb <- mco(yb, cbind(1, Xb)); s21 <- T * sum(bb$e^2) / (N - K - 1); th <- 1 - sqrt(s2e / s21)
  ys <- y - th * media_grupo(y, g); Xs <- cbind(1 - th, X - th * media_grupo(X, g))
  r <- mco(ys, Xs); list(b = r$b, V = sum(r$e^2) / (n - K - 1) * r$XtXi, th = th, s2e = s2e, s2u = (s21 - s2e) / T)
}
mc2e <- function(y, X, Z) {
  PZ <- Z %*% solve(crossprod(Z), crossprod(Z, X)); b <- as.vector(solve(crossprod(PZ, X), crossprod(PZ, y))); e <- y - X %*% b
  list(b = b, V = sum(e^2) / (length(y) - ncol(X)) * solve(crossprod(PZ)))
}
rezagar <- function(z, g, t, k = 1) { clave <- paste(g, t); z[match(paste(g, t - k), clave)] }
lcg_nuevo <- function(s) { e <- new.env(); e$x <- s; e }
lcg_u <- function(G) { G$x <- (16807 * G$x) %% 2147483647; G$x / 2147483647 }
lcg_normal <- function(G) { u1 <- lcg_u(G); u2 <- lcg_u(G); sqrt(-2 * log(u1)) * cos(2 * pi * u2) }
CARTERAS <- c("NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other",
              "S1V1", "S1V3", "S1V5", "S3V1", "S3V3", "S3V5", "S5V1", "S5V3", "S5V5",
              "S1M1", "S1M3", "S1M5", "S3M1", "S3M3", "S3M5", "S5M1", "S5M3", "S5M5")
panel_momentum <- function() {
  d <- read.csv("datos/internacional/french_mensual_1949_2017.csv"); Rb <- 100 * as.matrix(d[, CARTERAS]); R <- Rb - 100 * d$RF
  T <- nrow(R); N <- ncol(R); S <- matrix(NA, T, N); for (k in 13:T) S[k, ] <- colSums(Rb[(k - 12):(k - 2), , drop = FALSE])
  k <- 13:T
  list(y = as.vector(t(R[k, ])), x = as.vector(t(S[k, ])), cart = rep(0:(N - 1), length(k)), mes = rep(k - 1, each = N), f = as.character(d$dates))
}
