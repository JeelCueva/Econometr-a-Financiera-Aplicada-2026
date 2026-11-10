# Capítulo 19, Ejercicio 19.2: errores estándar en paneles: robustos, agrupados y de Driscoll-Kraay (Grunfeld)
source("codigo/R/cap19_funciones.R")
D <- grunfeld(); g <- D$g; t <- D$t; y <- D$y; X <- D$X; n <- nrow(X); K <- ncol(X); N <- length(unique(g)); T <- n / N
yw <- y - ave(y, g); Xw <- X - media_grupo(X, g); w <- mco(yw, Xw); b <- w$b; e <- w$e; A <- w$XtXi
E <- t(sapply(sort(unique(g)), function(c) e[g == c][order(t[g == c])]))
rho_t <- mean(sapply(1:N, function(i) cor(E[i, -1], E[i, -T]))); C <- cor(t(E)); rho_c <- mean(C[upper.tri(C)])
cat(sprintf("Efectos fijos de empresa: valor = %.4f; capital = %.4f\n", b[1], b[2]))
cat(sprintf("autocorrelación media de orden 1 de los residuos de cada empresa = %.4f\n", rho_t))
cat(sprintf("correlación media entre los residuos de dos empresas en el mismo año = %.4f\n", rho_c))
V <- list("clásicos" = v_clasica(Xw, e, A, n - N - K), "White (HC1)" = v_white(Xw, e, A, K + N),
          "agrupados por empresa" = v_cluster(Xw, e, A, g, K + N), "agrupados por año" = v_cluster(Xw, e, A, t, K + N),
          "dos vías" = v_dos_vias(Xw, e, A, g, t, K + N), "Driscoll-Kraay (L = 2)" = v_driscoll_kraay(Xw, e, A, t, 2))
cat("\n  valor     capital    t(valor)  t(capital)   error estándar\n")
for (nom in names(V)) { s <- sqrt(diag(V[[nom]]))
  cat(sprintf("  %.4f    %.4f    %7.2f   %7.2f      %s\n", s[1], s[2], b[1] / s[1], b[2] / s[2], nom)) }
cat(sprintf("\nvalor crítico al 5 %%: normal = 1,96; t con G - 1 = %d gl = %.4f\n", N - 1, qt(0.975, N - 1)))
y2 <- y - ave(y, g) - ave(y, t) + mean(y); X2 <- X - media_grupo(X, g) - media_grupo(X, t) + matrix(colMeans(X), n, K, byrow = TRUE)
w2 <- mco(y2, X2); s2 <- sqrt(diag(v_clasica(X2, w2$e, w2$XtXi, n - N - T + 1 - K))); s2c <- sqrt(diag(v_cluster(X2, w2$e, w2$XtXi, g, K + N + T - 1)))
cat(sprintf("\nEfectos fijos de empresa y año: valor = %.4f (clásico %.4f; agrupado %.4f); capital = %.4f (clásico %.4f; agrupado %.4f)\n", w2$b[1], s2[1], s2c[1], w2$b[2], s2[2], s2c[2]))
r <- y - X %*% w2$b; ef <- (ave(as.vector(r), t) - mean(r))[!duplicated(t)][order(unique(t))]
cat("efectos de año (desviaciones de la media): ", paste(sprintf("%.1f", ef), collapse = " "), "\n", sep = "")
