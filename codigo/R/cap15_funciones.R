# Capítulo 15: funciones propias para la estructura temporal de tasas de interés
TAU <- c(1, 2, 3, 5, 6, 11, 12, 36, 60, 120)              # vencimientos en meses
cargar <- function() {
  d <- read.csv("datos/internacional/tasas_cupon_cero_eeuu_mk.csv")
  list(fecha = d$fecha, Y = as.matrix(d[, paste0("r", TAU)]))
}
ns_cargas <- function(tau, lam) {                          # nivel, pendiente y curvatura
  x <- lam * tau; s <- (1 - exp(-x)) / x; cbind(1, s, s - exp(-x))
}
ns_ajuste <- function(Y, tau, lam) {                       # MCO de corte transversal, mes a mes
  X <- ns_cargas(tau, lam); B <- t(solve(crossprod(X), t(X) %*% t(Y))); A <- B %*% t(X)
  list(B = B, A = A, recm = sqrt(mean((Y - A)^2)))
}
ns_lambda <- function(Y, tau, malla) {                     # lambda común por búsqueda en malla
  sse <- sapply(malla, function(l) sum((Y - ns_ajuste(Y, tau, l)$A)^2)); list(lam = malla[which.min(sse)], sse = sse)
}
mco_nw <- function(y, X, L) {                              # MCO con errores de Newey-West
  X <- cbind(1, X); b <- solve(crossprod(X), crossprod(X, y)); e <- as.vector(y - X %*% b); n <- length(y)
  g <- X * e; S <- crossprod(g)
  for (j in 1:L) { G <- t(g[(j + 1):n, , drop = FALSE]) %*% g[1:(n - j), , drop = FALSE]; S <- S + (1 - j / (L + 1)) * (G + t(G)) }
  Xi <- solve(crossprod(X)); V <- Xi %*% S %*% Xi
  list(b = as.vector(b), ee = sqrt(diag(V)), r2 = 1 - sum(e^2) / sum((y - mean(y))^2), n = n)
}
