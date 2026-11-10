# Capítulo 20, Ejercicio 20.2: todos los predictores a la vez: MCO, ridge, lasso y red elástica fuera de muestra
source("codigo/R/cap20_funciones.R")
D <- cargar(); mes <- D$mes; X <- D$X[, IDX_MULT]; y <- D$y; n <- length(y); i0 <- which(mes == "1957-01")
fb <- sapply(i0:n, function(i) mean(y[1:(i - 1)])); yo <- y[i0:n]
anios <- seq(i0, n, by = 12); M <- list("MCO (todos)" = NA, "ridge" = 0, "red elástica (0,5)" = 0.5, "lasso" = 1)
F <- lapply(M, function(z) numeric(n - i0 + 1)); sel <- numeric(length(VARS_MULT)); nsel <- c(); lams <- lapply(M, function(z) c())
for (a in anios) { tr <- 1:(a - 1); te <- a:min(a + 11, n); k0 <- a - i0 + 1; kk <- k0:(k0 + length(te) - 1)
  S <- estandarizar(X[tr, ], X[te, , drop = FALSE]); my <- mean(y[tr])
  Z <- cbind(1, S$tr); b <- solve(crossprod(Z), crossprod(Z, y[tr])); F[["MCO (todos)"]][kk] <- b[1] + S$te %*% b[-1]
  for (nom in names(M)[-1]) { r <- ajustar_validado(X[tr, ], y[tr], M[[nom]]); lams[[nom]] <- c(lams[[nom]], r$lam); F[[nom]][kk] <- my + S$te %*% r$b
    if (nom == "lasso") { sel <- sel + (r$b != 0); nsel <- c(nsel, sum(r$b != 0)) } } }
cat(sprintf("Predictores: %s (de = dp - ep y tms = lty - tbl se omiten por colinealidad exacta)\n", paste(VARS_MULT, collapse = ", ")))
cat(sprintf("Fuera de muestra %s a %s; reestimación anual (%d veces); lambda elegido con los últimos 120 meses de cada muestra\n", mes[i0], mes[n], length(anios)))
cat("R2 fuera (%)   R2 fuera CT (%)    CW     p(CW)   lambda mediano   modelo\n")
for (nom in names(M)) { cw <- clark_west(yo, F[[nom]], fb); lm_ <- if (length(lams[[nom]])) sprintf("%.4f", median(lams[[nom]])) else "  -"
  cat(sprintf("%11.2f  %14.2f  %7.2f  %7.4f   %12s   %s\n", 100 * r2_os(yo, F[[nom]], fb), 100 * r2_os(yo, pmax(F[[nom]], 0), fb), cw[1], cw[2], lm_, nom)) }
cat(sprintf("\nLasso: número medio de predictores seleccionados = %.2f; reestimaciones sin ningún predictor = %d de %d\n", mean(nsel), sum(nsel == 0), length(nsel)))
cat("fracción de reestimaciones en que el lasso selecciona cada predictor:\n")
cat("  ", paste(sprintf("%s=%.2f", VARS_MULT, sel / length(anios)), collapse = "  "), "\n", sep = "")
cat(sprintf("\ndesviación estándar de los pronósticos: media histórica = %.4f; MCO = %.4f; ridge = %.4f; lasso = %.4f\n", sd(fb), sd(F[["MCO (todos)"]]), sd(F[["ridge"]]), sd(F[["lasso"]])))
