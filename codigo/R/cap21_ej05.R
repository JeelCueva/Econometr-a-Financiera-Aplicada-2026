# Capítulo 21, Etapa 5: asignación: 1/N frente a mínima varianza fuera de muestra
source("codigo/R/cap21_funciones.R")
D <- cargar(); f <- D$fecha; A <- D$R[, 1:4]; n <- nrow(A); V <- 250; paso <- 21
H <- array(0, c(n, 4, 4)); H[1, , ] <- crossprod(A[1:30, ]) / 30
for (t in 2:n) H[t, , ] <- 0.94 * H[t - 1, , ] + 0.06 * outer(A[t - 1, ], A[t - 1, ])
reb <- seq(V + 1, n, by = paso); noms <- c("1/N", "mínima varianza (muestral, 250 días)", "mínima varianza (EWMA)")
W <- list(t(sapply(reb, function(t) rep(0.25, 4))), t(sapply(reb, function(t) min_varianza(cov(A[(t - V):(t - 1), ])))), t(sapply(reb, function(t) min_varianza(H[t, , ]))))
cat(sprintf("Evaluación: %s a %s (%d días); rebalanceo cada %d días (%d veces); pesos constantes entre rebalanceos\n", f[V + 1], f[n], n - V, paso, length(reb)))
cat("\nmedia anual (%)   vol. anual (%)   Sharpe anual   ee(Sharpe)   máx. caída (%)   rotación media   peso mínimo medio   estrategia\n")
RP <- list()
for (i in 1:3) { Ws <- W[[i]]; rp <- numeric(n - V)
  for (k in seq_along(reb)) { t <- reb[k]; fin <- min(t + paso - 1, n); rp[(t - V):(fin - V)] <- A[t:fin, , drop = FALSE] %*% Ws[k, ] }
  RP[[i]] <- rp; sr <- mean(rp) / sd(rp); T <- length(rp); ee <- sqrt((1 + sr^2 / 2) / T) * sqrt(252); rot <- mean(rowSums(abs(diff(Ws))))
  cat(sprintf("%14.2f  %15.2f  %13.3f  %11.3f  %14.2f  %14.3f  %17.3f   %s\n", 252 * mean(rp), sqrt(252) * sd(rp), sqrt(252) * sr, ee, max_caida(rp), rot, mean(apply(Ws, 1, min)), noms[i])) }
Wm <- W[[2]]
cat("\npesos medios de mínima varianza (muestral): ", paste(sprintf("%s = %.3f", NOMBRES[1:4], colMeans(Wm)), collapse = "  "), "\n", sep = "")
cat(sprintf("rebalanceos con alguna posición corta: %d de %d\n", sum(apply(Wm, 1, min) < 0), length(reb)))
cat(sprintf("diferencia de varianzas diarias (mínima varianza - 1/N) = %.4f; correlación entre ambas estrategias = %.4f\n", var(RP[[2]]) - var(RP[[1]]), cor(RP[[1]], RP[[2]])))
