# Capítulo 19, Ejercicio 19.5: efectos fijos de cartera y de mes: momentum de series de tiempo frente a momentum de corte transversal
source("codigo/R/cap19_funciones.R")
P <- panel_momentum(); y <- P$y; x <- P$x; c <- P$cart; m <- P$mes; n <- length(y); N <- length(unique(c)); T <- length(unique(m))
quitar <- function(z, cartera, mes) {
  if (cartera && mes) return(z - ave(z, c) - ave(z, m) + mean(z))
  if (cartera) return(z - ave(z, c)); if (mes) return(z - ave(z, m)); z - mean(z) }
cat("especificación                    pendiente   ee(mes)   ee(dos vías)   t(dos vías)   varianza de x que usa\n")
esp <- list(c("agrupado (sin efectos)", 0, 0), c("efectos de cartera", 1, 0), c("efectos de mes", 0, 1), c("cartera y mes", 1, 1)); res <- c()
for (s in esp) { cc <- s[2] == "1"; mm <- s[3] == "1"; yt <- quitar(y, cc, mm); xt <- quitar(x, cc, mm); X <- matrix(xt)
  r <- mco(yt, X); k <- 1 + (N - 1) * cc + (T - 1) * mm + 1
  sm <- sqrt(v_cluster(X, r$e, r$XtXi, m, k)[1, 1]); sd2 <- sqrt(v_dos_vias(X, r$e, r$XtXi, c, m, k)[1, 1]); res[s[1]] <- r$b
  cat(sprintf("  %s%s  %9.5f  %8.5f  %10.5f  %10.2f      %.4f\n", s[1], strrep(" ", 30 - nchar(s[1])), r$b, sm, sd2, r$b / sd2, sum(xt^2) / sum((x - mean(x))^2))) }
xm <- ave(x, m); ym <- ave(y, m); idx <- !duplicated(m); bt <- mco(ym[idx], cbind(1, xm[idx]))$b
cat(sprintf("\nRegresión de series de tiempo del promedio de las 30 carteras sobre su propio x promedio: pendiente = %.5f\n", bt[2]))
w <- c(sum(quitar(x, FALSE, TRUE)^2), sum((xm - mean(x))^2))
cat(sprintf("Descomposición de la varianza de x: dentro de cada mes = %.4f; entre meses = %.4f\n", w[1] / sum(w), w[2] / sum(w)))
cat(sprintf("Pendiente agrupada = %.4f x %.5f + %.4f x %.5f = %.5f\n", w[1] / sum(w), res["efectos de mes"], w[2] / sum(w), bt[2], w[1] / sum(w) * res["efectos de mes"] + w[2] / sum(w) * bt[2]))
cat(sprintf("\nInterpretación económica de la pendiente con efectos de mes: 10 puntos más de rendimiento pasado -> %.3f puntos más de exceso de rendimiento mensual\n", 10 * res["efectos de mes"]))
