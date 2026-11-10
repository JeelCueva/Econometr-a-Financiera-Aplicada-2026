# Capítulo 14, Ejercicio 14.1: CAPM frente a Fama-French en las 9 carteras de tamaño y valor
source("codigo/R/cap14_funciones.R")
D <- datos(); R <- D$R[, VALOR]; T <- nrow(R)
cat(sprintf("Muestra: %s a %s, T = %d meses; excesos de rendimiento en %% mensual\n\n", D$fecha[1], D$fecha[T], T))
c1 <- regs_ts(R, D$F[, "MktRF", drop = FALSE]); c3 <- regs_ts(R, D$F[, c("MktRF", "SMB", "HML")])
cat("Cartera   media   | CAPM: alfa  t(alfa)  beta   R2  | FF3: alfa  t(alfa)  b_Mkt   b_SMB   b_HML   R2\n")
for (i in seq_along(VALOR))
  cat(sprintf("%-7s %7.3f   | %8.3f %7.2f %6.2f %5.2f | %8.3f %7.2f %6.2f %7.2f %7.2f %5.2f\n", VALOR[i], mean(R[, i]),
              c1$alfa[i], c1$t_alfa[i], c1$beta[i, 1], c1$r2[i], c3$alfa[i], c3$t_alfa[i], c3$beta[i, 1], c3$beta[i, 2], c3$beta[i, 3], c3$r2[i]))
cat(sprintf("\npromedio de |alfa|: CAPM = %.3f; FF3 = %.3f (%% mensual)\n", mean(abs(c1$alfa)), mean(abs(c3$alfa))))
cat(sprintf("diferencia de medias S1V5 - S1V1 = %.3f; de alfas CAPM = %.3f; de alfas FF3 = %.3f\n",
            mean(R[, "S1V5"]) - mean(R[, "S1V1"]), c1$alfa[3] - c1$alfa[1], c3$alfa[3] - c3$alfa[1]))
