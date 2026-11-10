# Capítulo 19, Ejercicio 19.1: la ecuación de inversión de Grunfeld con MCO agrupado, between, efectos fijos y aleatorios
source("codigo/R/cap19_funciones.R")
D <- grunfeld(); g <- D$g; t <- D$t; y <- D$y; X <- D$X; n <- nrow(X); K <- ncol(X); N <- length(unique(g)); T <- n / N
cat(sprintf("Panel balanceado: N = %d empresas, T = %d años (%d-%d), n = %d\n", N, T, min(t), max(t), n))
cat(sprintf("medias: inversión = %.2f; valor = %.2f; capital = %.2f\n", mean(y), mean(X[, 1]), mean(X[, 2])))
fila <- function(nom, b, V, j0 = 1) { ee <- sqrt(diag(V))
  cat(sprintf("  %s%s valor %8.4f (%.4f)   capital %8.4f (%.4f)\n", nom, strrep(" ", 22 - nchar(nom)), b[j0], ee[j0], b[j0 + 1], ee[j0 + 1])) }
Z <- cbind(1, X); p <- mco(y, Z); fila("MCO agrupado", p$b, v_clasica(Z, p$e, p$XtXi, n - K - 1), 2)
idx <- !duplicated(g); yb <- ave(y, g)[idx]; Xb <- media_grupo(X, g)[idx, ]
Zb <- cbind(1, Xb); bt <- mco(yb, Zb); fila("between", bt$b, v_clasica(Zb, bt$e, bt$XtXi, N - K - 1), 2)
yw <- y - ave(y, g); Xw <- X - media_grupo(X, g); w <- mco(yw, Xw); Vw <- v_clasica(Xw, w$e, w$XtXi, n - N - K); fila("efectos fijos (within)", w$b, Vw)
re <- efectos_aleatorios(y, X, g); fila("efectos aleatorios", re$b, re$V, 2)
o <- order(g, t); gg <- g[o]; yy <- y[o]; XX <- X[o, ]; m <- gg[-1] == gg[-n]
dy <- diff(yy)[m]; dX <- (XX[-1, ] - XX[-n, ])[m, ]; fd <- mco(dy, dX); fila("primeras diferencias", fd$b, v_clasica(dX, fd$e, fd$XtXi, length(dy) - K))
cat(sprintf("\nefectos aleatorios: sigma_u = %.4f; sigma_e = %.4f; theta = %.4f; constante = %.4f\n", sqrt(re$s2u), sqrt(re$s2e), re$th, re$b[1]))
alfa <- (ave(y, g) - media_grupo(X, g) %*% w$b)[idx]
cat("efectos fijos estimados por empresa: ", paste(sprintf("%.1f", alfa), collapse = " "), "\n", sep = "")
Fe <- ((sum(p$e^2) - sum(w$e^2)) / (N - 1)) / (sum(w$e^2) / (n - N - K))
cat(sprintf("F de efectos individuales = %.4f con (%d, %d) gl; valor p = %.2e\n", Fe, N - 1, n - N - K, pf(Fe, N - 1, n - N - K, lower.tail = FALSE)))
d <- w$b - re$b[-1]; H <- drop(t(d) %*% solve(Vw - re$V[-1, -1], d))
cat(sprintf("Hausman (efectos fijos frente a aleatorios) = %.4f; valor p chi2(2) = %.4f\n", H, pchisq(H, 2, lower.tail = FALSE)))
cat(sprintf("R2 within = %.4f; R2 del MCO agrupado = %.4f\n", 1 - sum(w$e^2) / sum(yw^2), 1 - sum(p$e^2) / sum((y - mean(y))^2)))
