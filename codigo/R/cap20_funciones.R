# Capítulo 20: funciones propias para aprendizaje automático en finanzas
VARS <- c("dp", "ep", "de", "cape", "lty", "tbl", "tms", "infl", "mom12", "vol12")
cargar <- function() { d <- read.csv("datos/internacional/predictores_mensuales_1927_2018.csv"); n <- nrow(d)
  list(mes = d$mes[-1], X = as.matrix(d[-n, VARS]), y = d$exceso[-1]) }
dpob <- function(X) sqrt(colMeans(sweep(X, 2, colMeans(X))^2))
estandarizar <- function(Xtr, Xte) { m <- colMeans(Xtr); s <- dpob(Xtr)
  list(tr = sweep(sweep(Xtr, 2, m), 2, s, "/"), te = sweep(sweep(Xte, 2, m), 2, s, "/")) }
red_elastica <- function(X, y, lam, alfa, b0 = NULL, tol = 1e-9, maxit = 10000) {
  n <- nrow(X); p <- ncol(X); b <- if (is.null(b0)) numeric(p) else b0; r <- as.vector(y - X %*% b)
  for (it in 1:maxit) { cambio <- 0
    for (j in 1:p) { z <- sum(X[, j] * r) / n + b[j]
      nb <- sign(z) * max(abs(z) - lam * alfa, 0) / (1 + lam * (1 - alfa))
      if (nb != b[j]) { r <- r - X[, j] * (nb - b[j]); cambio <- max(cambio, abs(nb - b[j])); b[j] <- nb } }
    if (cambio < tol) break }
  b
}
malla_lambda <- function(X, y, alfa, k = 30) max(abs(crossprod(X, y))) / (length(y) * max(alfa, 0.01)) * 10^seq(0, -4, length.out = k)
ajustar_validado <- function(X, y, alfa, nval = 120) {
  n <- nrow(X); ia <- 1:(n - nval); iv <- (n - nval + 1):n
  S <- estandarizar(X[ia, , drop = FALSE], X[iv, , drop = FALSE]); ma <- mean(y[ia]); malla <- malla_lambda(S$tr, y[ia] - ma, alfa)
  b <- NULL; ecm <- numeric(length(malla))
  for (k in seq_along(malla)) { b <- red_elastica(S$tr, y[ia] - ma, malla[k], alfa, b); ecm[k] <- mean((y[iv] - ma - S$te %*% b)^2) }
  lam <- malla[which.min(ecm)]; Xs <- estandarizar(X, X)$tr
  list(lam = lam, b = red_elastica(Xs, y - mean(y), lam, alfa))
}
r2_os <- function(y, f, fb) 1 - sum((y - f)^2) / sum((y - fb)^2)
clark_west <- function(y, f, fb) { a <- (y - fb)^2 - ((y - f)^2 - (fb - f)^2); t <- mean(a) / (sd(a) / sqrt(length(a))); c(t, pnorm(t, lower.tail = FALSE)) }
# de = dp - ep y tms = lty - tbl son combinaciones lineales exactas: los modelos multivariados usan 8 predictores
VARS_MULT <- c("dp", "ep", "cape", "lty", "tbl", "infl", "mom12", "vol12")
IDX_MULT <- match(VARS_MULT, VARS)
logit <- function(y, X, tol = 1e-12, maxit = 100) {
  b <- numeric(ncol(X))
  for (it in 1:maxit) { p <- 1 / (1 + exp(-X %*% b)); g <- crossprod(X, y - p); H <- crossprod(X * as.vector(p * (1 - p)), X); d <- solve(H, g); b <- b + as.vector(d)
    if (max(abs(d)) < tol) break }
  p <- as.vector(1 / (1 + exp(-X %*% b))); H <- crossprod(X * (p * (1 - p)), X)
  list(b = b, V = solve(H), ll = sum(y * log(p) + (1 - y) * log(1 - p)))
}
auc <- function(y, s) { r <- rank(s); n1 <- sum(y); n0 <- length(y) - n1; (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
mejor_corte <- function(X, y, minhoja) {
  n <- nrow(X); mejor <- list(sse = Inf, j = NULL, u = NULL)
  for (j in 1:ncol(X)) { o <- order(X[, j], method = "radix"); xs <- X[o, j]; ys <- y[o]; c1 <- cumsum(ys); c2 <- cumsum(ys^2)
    for (k in minhoja:(n - minhoja)) { if (xs[k] == xs[k + 1]) next
      sl <- c1[k]; sr <- c1[n] - c1[k]; sse <- (c2[k] - sl^2 / k) + (c2[n] - c2[k] - sr^2 / (n - k))
      if (sse < mejor$sse) mejor <- list(sse = sse, j = j, u = (xs[k] + xs[k + 1]) / 2) } }
  mejor
}
arbol <- function(X, y, prof, minhoja = 24) {
  if (prof == 0 || length(y) < 2 * minhoja) return(list(hoja = mean(y), n = length(y)))
  m <- mejor_corte(X, y, minhoja); if (is.null(m$j)) return(list(hoja = mean(y), n = length(y)))
  iz <- X[, m$j] <= m$u
  list(var = m$j, umbral = m$u, n = length(y), izq = arbol(X[iz, , drop = FALSE], y[iz], prof - 1, minhoja), der = arbol(X[!iz, , drop = FALSE], y[!iz], prof - 1, minhoja))
}
predecir <- function(t, X) { if (!is.null(t$hoja)) return(rep(t$hoja, nrow(X)))
  iz <- X[, t$var] <= t$umbral; out <- numeric(nrow(X))
  if (any(iz)) out[iz] <- predecir(t$izq, X[iz, , drop = FALSE]); if (any(!iz)) out[!iz] <- predecir(t$der, X[!iz, , drop = FALSE]); out }
imprimir_arbol <- function(t, nombres, sangria = "  ") {
  if (!is.null(t$hoja)) { cat(sprintf("%shoja: media = %.4f (n = %d)\n", sangria, t$hoja, t$n)); return(invisible()) }
  cat(sprintf("%ssi %s <= %.4f:\n", sangria, nombres[t$var], t$umbral)); imprimir_arbol(t$izq, nombres, paste0(sangria, "    "))
  cat(sprintf("%ssi %s > %.4f:\n", sangria, nombres[t$var], t$umbral)); imprimir_arbol(t$der, nombres, paste0(sangria, "    "))
}
