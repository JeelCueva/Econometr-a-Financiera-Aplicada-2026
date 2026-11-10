# Capítulo 16, Ejercicio 16.3: los anuncios del FOMC y el S&P 500 (40 reuniones programadas, 2014-2018)
source("codigo/R/cap16_funciones.R")
s <- read.csv("datos/internacional/sp500_diario_yahoo.csv"); P <- s$Adj.Close
f <- s$Date[-1]; r <- 100 * (P[-1] / P[-length(P)] - 1)
ev <- read.csv("datos/internacional/fomc_anuncios_2014_2018.csv")$fecha_anuncio
T0 <- match(ev, f); N <- length(T0)
res <- lapply(T0, function(t0) rendimientos_normales(r, r, t0, modelo = "media"))
AR <- t(sapply(res, `[[`, "AR")); V <- t(sapply(res, function(x) diag(x$V))); gl <- sapply(res, `[[`, "gl")
cat(sprintf("N = %d anuncios, de %s a %s; modelo de rendimiento medio, estimación [-250, -11]\n\n", N, ev[1], ev[N]))
cat(" tau    CAAR      J1    Patell    BMP    signo  positivos\n")
for (k in 1:11) { p <- pruebas(AR[, k], V[, k], gl)
  cat(sprintf(" %3d %8.4f %7.2f %7.2f %7.2f %7.2f   %2d/%d\n", k - 6, p$CAAR, p$J1, p$Patell, p$BMP, p$signo, p$pos, N)) }
caar <- function(t1, t2) { k <- (t1 + 6):(t2 + 6); c <- rowSums(AR[, k, drop = FALSE]); v <- sapply(res, function(x) sum(x$V[k, k]))
  sc <- c / sqrt(v); c(mean(c), mean(c) / sqrt(sum(v) / N^2), mean(sc) / (sd(sc) / sqrt(N))) }
cat("\nventana     CAAR     J1     BMP\n")
for (w in list(c(-1, 0), c(-1, 1), c(-5, -1), c(0, 5), c(-5, 5))) { x <- caar(w[1], w[2])
  cat(sprintf("(%2d,%+d)  %8.4f %6.2f %7.2f\n", w[1], w[2], x[1], x[2], x[3])) }
A <- t(sapply(res, function(x) c(x$e, x$AR[6])))
cat(sprintf("\nPrueba de rangos de Corrado para el día 0: %.2f\n", corrado(A)))
i <- which(f >= "2014-01-02" & f <= "2018-12-31"); d0 <- as.numeric(i %in% T0); d1 <- as.numeric(i %in% (T0 - 1))
X <- cbind(1, d1, d0); y <- r[i]; b <- solve(crossprod(X), crossprod(X, y)); e <- as.vector(y - X %*% b)
XtXi <- solve(crossprod(X)); Vw <- XtXi %*% crossprod(X * e) %*% XtXi
cat(sprintf("\nRegresión 2014-2018 (n = %d): r_t = c + d1 D(-1) + d0 D(0); errores de White (HC0)\n", length(i)))
for (k in 1:3) cat(sprintf("  %s %8.4f  ee %.4f  t %6.2f\n", c("constante ", "día -1    ", "día 0     ")[k], b[k], sqrt(Vw[k, k]), b[k] / sqrt(Vw[k, k])))
cat(sprintf("Desv. est. en días de anuncio = %.4f; en días sin anuncio (ni víspera) = %.4f\n", sd(r[T0]), sd(y[d0 + d1 == 0])))
