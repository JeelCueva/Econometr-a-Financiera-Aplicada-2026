# Capítulo 19, Ejercicio 19.4: ¿qué errores estándar? Petersen (2009) con un panel de 30 carteras de French
source("codigo/R/cap19_funciones.R")
P <- panel_momentum(); y <- P$y; x <- P$x; c <- P$cart; m <- P$mes; f <- P$f; n <- length(y); N <- length(unique(c)); T <- length(unique(m))
cat(sprintf("Panel: N = %d carteras, T = %d meses (%s a %s), n = %d\n", N, T, f[min(m) + 1], f[max(m) + 1], n))
cat(sprintf("y = exceso de rendimiento del mes t (%%); x = rendimiento acumulado de t-12 a t-2 (%%): media = %.4f; desv. est. = %.4f\n", mean(x), sd(x)))
X <- cbind(1, x); r <- mco(y, X); b <- r$b; e <- r$e; A <- r$XtXi
E <- matrix(e, T, N, byrow = TRUE); Xs <- matrix(x, T, N, byrow = TRUE)
corr_media <- function(M, eje) if (eje == "tiempo") mean(sapply(1:N, function(j) cor(M[-1, j], M[-T, j]))) else { C <- cor(M); mean(C[upper.tri(C)]) }
cat(sprintf("\nCorrelación media de los residuos entre carteras en el mismo mes = %.4f; de x = %.4f\n", corr_media(E, "corte"), corr_media(Xs, "corte")))
cat(sprintf("Autocorrelación media de orden 1 de los residuos de cada cartera = %.4f; de x = %.4f\n", corr_media(E, "tiempo"), corr_media(Xs, "tiempo")))
cat(sprintf("\nMCO agrupado: constante = %.4f; pendiente = %.5f\n", b[1], b[2]))
V <- list("clásicos" = v_clasica(X, e, A, n - 2), "White (HC1)" = v_white(X, e, A, 2), "agrupados por cartera" = v_cluster(X, e, A, c, 2),
          "agrupados por mes" = v_cluster(X, e, A, m, 2), "dos vías" = v_dos_vias(X, e, A, c, m, 2), "Driscoll-Kraay (L = 12)" = v_driscoll_kraay(X, e, A, m, 12))
cat("   ee(pendiente)   t      cociente con el clásico   método\n")
s0 <- sqrt(V[["clásicos"]][2, 2])
for (nom in names(V)) { s <- sqrt(V[[nom]][2, 2]); cat(sprintf("   %.5f   %7.2f   %10.3f                  %s\n", s, b[2] / s, s / s0, nom)) }
Y <- matrix(y, T, N, byrow = TRUE); B <- sapply(1:T, function(k) coef(lm(Y[k, ] ~ Xs[k, ]))[2])
fm <- mean(B); sfm <- sd(B) / sqrt(T); h <- B - fm; S <- sum(h^2) / T
for (j in 1:12) S <- S + 2 * (1 - j / 13) * sum(h[(j + 1):T] * h[1:(T - j)]) / T
snw <- sqrt(S / T)
cat(sprintf("   %.5f   %7.2f   %10.3f                  Fama-MacBeth (pendiente media = %.5f)\n", sfm, fm / sfm, sfm / s0, fm))
cat(sprintf("   %.5f   %7.2f   %10.3f                  Fama-MacBeth con Newey-West (12 rezagos)\n", snw, fm / snw, snw / s0))
cat(sprintf("\nautocorrelación de orden 1 de las pendientes mensuales de Fama-MacBeth = %.4f\n", cor(B[-1], B[-T])))
