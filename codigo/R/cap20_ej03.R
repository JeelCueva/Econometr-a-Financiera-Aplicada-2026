# Capítulo 20, Ejercicio 20.3: combinación de pronósticos, componentes principales y valor económico
source("codigo/R/cap20_funciones.R")
D <- cargar(); mes <- D$mes; X <- D$X; y <- D$y; n <- length(y); i0 <- which(mes == "1957-01"); yo <- y[i0:n]; m <- n - i0 + 1
fb <- sapply(i0:n, function(i) mean(y[1:(i - 1)]))
U <- sapply(seq_along(VARS), function(j) sapply(i0:n, function(i) { Z <- cbind(1, X[1:(i - 1), j]); b <- solve(crossprod(Z), crossprod(Z, y[1:(i - 1)])); b[1] + b[2] * X[i, j] }))
F <- list("combinación (media de 10)" = rowMeans(U), "combinación (mediana de 10)" = apply(U, 1, median))
Xm <- X[, IDX_MULT]
for (q in 1:3) { f <- numeric(m)
  for (a in seq(i0, n, by = 12)) { tr <- 1:(a - 1); te <- a:min(a + 11, n); S <- estandarizar(Xm[tr, ], Xm[te, , drop = FALSE])
    V <- eigen(crossprod(S$tr) / length(tr), symmetric = TRUE)$vectors[, 1:q, drop = FALSE]
    Z <- cbind(1, S$tr %*% V); b <- solve(crossprod(Z), crossprod(Z, y[tr])); f[(a - i0 + 1):(a - i0 + length(te))] <- b[1] + S$te %*% V %*% b[-1] }
  F[[sprintf("componentes principales (k = %d)", q)]] <- f }
cat(sprintf("Fuera de muestra %s a %s (%d meses)\n", mes[i0], mes[n], m))
cat("R2 fuera (%)    CW     p(CW)   modelo\n")
for (nom in names(F)) { cw <- clark_west(yo, F[[nom]], fb); cat(sprintf("%11.2f  %6.2f  %7.4f   %s\n", 100 * r2_os(yo, F[[nom]], fb), cw[1], cw[2], nom)) }
cat(sprintf("\ndesviación estándar de los pronósticos: combinación = %.4f; media histórica = %.4f\n", sd(F[[1]]), sd(fb)))
cat(sprintf("correlación entre la combinación y el exceso realizado = %.4f\n", cor(F[[1]], yo)))
gam <- 3; s2 <- sapply(i0:n, function(i) var(y[(i - 60):(i - 1)])); rf <- X[i0:n, "tbl"] / 12
eqc <- function(f) { w <- pmin(pmax(100 * f / (gam * s2), 0), 1.5); rp <- rf + w * yo; list(ce = 12 * (mean(rp) - 0.5 * gam * var(rp) / 100), w = w) }
e0 <- eqc(fb)
cat("\nInversionista media-varianza (aversión 3, varianza de 60 meses, peso en acciones entre 0 y 150 %):\n")
cat(sprintf("  equivalente cierto anual con la media histórica = %.3f %%; peso medio en acciones = %.3f\n", e0$ce, mean(e0$w)))
for (nom in c("combinación (media de 10)", "componentes principales (k = 1)")) { e <- eqc(F[[nom]])
  cat(sprintf("  %s: equivalente cierto = %.3f %%; ganancia = %.3f puntos por año; peso medio = %.3f\n", nom, e$ce, e$ce - e0$ce, mean(e$w))) }
e <- eqc(U[, which(VARS == "dp")]); cat(sprintf("  solo dp: equivalente cierto = %.3f %%; ganancia = %.3f puntos por año; peso medio = %.3f\n", e$ce, e$ce - e0$ce, mean(e$w)))
