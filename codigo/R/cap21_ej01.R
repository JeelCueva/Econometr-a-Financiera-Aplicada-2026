# Capítulo 21, Etapa 1: los datos y sus hechos estilizados
source("codigo/R/cap21_funciones.R")
D <- cargar(); f <- D$fecha; R <- D$R; n <- length(f)
cat(sprintf("Rendimientos simples diarios (%%), %s a %s; n = %d\n\n", f[1], f[n], n))
cat("serie    media anual   vol. anual   asimetría   curtosis    JB        LB(10) r   p      LB(10) r2   p      peor día\n")
for (j in seq_along(NOMBRES)) { x <- R[, j]; s <- asimetria(x); k <- curtosis(x); jb <- n / 6 * (s^2 + (k - 3)^2 / 4)
  l1 <- ljung_box(x, 10); l2 <- ljung_box(x^2, 10); i <- which.min(x)
  cat(sprintf("%-6s  %10.2f  %10.2f  %10.3f  %9.2f  %9.1f  %9.2f  %6.4f  %9.2f  %6.4f   %7.2f (%s)\n", NOMBRES[j], 252 * mean(x), sqrt(252) * sd(x), s, k, jb, l1[1], l1[2], l2[1], l2[2], x[i], f[i])) }
cat("\nMatriz de correlaciones:\n"); C <- cor(R); cat("        ", paste(sprintf("%7s", NOMBRES), collapse = " "), "\n", sep = "")
for (j in seq_along(NOMBRES)) cat(sprintf("%-6s  %s\n", NOMBRES[j], paste(sprintf("%7.3f", C[j, ]), collapse = " ")))
cat(sprintf("\nValor crítico de JB al 1 %% (chi2 con 2 gl) = %.2f; de LB(10) = %.2f\n", qchisq(0.99, 2), qchisq(0.99, 10)))
