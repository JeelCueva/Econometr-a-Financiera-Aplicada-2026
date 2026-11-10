# Capítulo 19, Ejercicio 19.3: paneles dinámicos: el sesgo de Nickell y el estimador de Anderson-Hsiao
source("codigo/R/cap19_funciones.R")
D <- grunfeld(); g <- D$g; t <- D$t; y <- log(D$y); x <- log(D$X[, 1])
y1 <- rezagar(y, g, t, 1); y2 <- rezagar(y, g, t, 2); x1 <- rezagar(x, g, t, 1)
cat("Modelo: ln I_it = rho ln I_i,t-1 + beta ln F_it + alfa_i + e_it  (Grunfeld, 1935-1954)\n")
ok <- !is.na(y1); n1 <- sum(ok); Z <- cbind(1, y1[ok], x[ok]); p <- mco(y[ok], Z); s <- sqrt(diag(v_clasica(Z, p$e, p$XtXi, n1 - 3)))
cat(sprintf("  MCO agrupado:        rho = %.4f (%.4f); beta = %.4f (%.4f); n = %d\n", p$b[2], s[2], p$b[3], s[3], n1))
gg <- g[ok]; N <- length(unique(gg)); W <- cbind(y1[ok], x[ok])
w <- mco(y[ok] - ave(y[ok], gg), W - media_grupo(W, gg)); sw <- sqrt(diag(v_clasica(W, w$e, w$XtXi, n1 - N - 2)))
cat(sprintf("  efectos fijos:       rho = %.4f (%.4f); beta = %.4f (%.4f)\n", w$b[1], sw[1], w$b[2], sw[2]))
ok2 <- !is.na(y2); dy <- (y - y1)[ok2]; dy1 <- (y1 - y2)[ok2]; dx <- (x - x1)[ok2]
ah <- mc2e(dy, cbind(dy1, dx), cbind(y2[ok2], dx)); sah <- sqrt(diag(ah$V))
cat(sprintf("  Anderson-Hsiao (VI): rho = %.4f (%.4f); beta = %.4f (%.4f); n = %d; instrumento: ln I_i,t-2\n", ah$b[1], sah[1], ah$b[2], sah[2], sum(ok2)))
cat(sprintf("  correlación entre el instrumento y el regresor endógeno = %.4f\n", cor(y2[ok2], dy1)))
T <- 19; cat(sprintf("\nSesgo aproximado de Nickell con T = %d y rho = %.2f: -(1 + rho)/(T - 1) = %.4f\n", T, w$b[1], -(1 + w$b[1]) / (T - 1)))
G <- lcg_nuevo(2019); REP <- 1000; N <- 10; T <- 20; rho <- 0.7; est <- matrix(0, REP, 2)
for (r in 1:REP) {
  a <- sapply(1:N, function(i) lcg_normal(G)); Y <- matrix(0, N, T + 50)
  for (i in 1:N) { Y[i, 1] <- a[i] / (1 - rho); for (k in 2:(T + 50)) Y[i, k] <- a[i] + rho * Y[i, k - 1] + lcg_normal(G) }
  Y <- Y[, 51:(T + 50)]; yy <- Y[, -1]; yl <- Y[, -T]
  yyc <- yy - rowMeans(yy); ylc <- yl - rowMeans(yl); fe <- sum(yyc * ylc) / sum(ylc^2)
  d <- as.vector(t(Y[, 3:T] - Y[, 2:(T - 1)])); dl <- as.vector(t(Y[, 2:(T - 1)] - Y[, 1:(T - 2)])); l2 <- as.vector(t(Y[, 1:(T - 2)]))
  est[r, ] <- c(fe, sum(l2 * d) / sum(l2 * dl))
}
cat(sprintf("Simulación (%d paneles con N = %d, T = %d, rho = %.1f):\n", REP, N, T, rho))
cat(sprintf("  efectos fijos:  media = %.4f; sesgo = %.4f; desv. est. = %.4f; fórmula -(1+rho)/(T-1) = %.4f\n", mean(est[, 1]), mean(est[, 1]) - rho, sd(est[, 1]), -(1 + rho) / (T - 2)))
cat(sprintf("  Anderson-Hsiao: mediana = %.4f; sesgo de la mediana = %.4f; rango intercuartílico = %.4f\n", median(est[, 2]), median(est[, 2]) - rho, quantile(est[, 2], 0.75, names = FALSE) - quantile(est[, 2], 0.25, names = FALSE)))
