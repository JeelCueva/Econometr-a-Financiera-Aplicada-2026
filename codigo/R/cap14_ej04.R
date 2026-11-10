# Capítulo 14, Ejercicio 14.4: la prueba GRS como comparación de razones de Sharpe
source("codigo/R/cap14_funciones.R")
D <- datos(); R <- D$R[, VALOR]; F <- D$F
for (mn in c("CAPM", "FF3", "Carhart")) {
  fac <- switch(mn, CAPM = "MktRF", FF3 = c("MktRF", "SMB", "HML"), Carhart = c("MktRF", "SMB", "HML", "Mom"))
  sf <- sr2(F[, fac, drop = FALSE]); sa <- sr2(cbind(F[, fac, drop = FALSE], R)); g <- grs(R, F[, fac, drop = FALSE])
  cat(sprintf("%-8s SR factores = %.4f (anual %.3f); SR factores + 9 carteras = %.4f (anual %.3f)\n", mn, sqrt(sf), sqrt(12 * sf), sqrt(sa), sqrt(12 * sa)))
  cat(sprintf("         SR2 todos - SR2 factores = %.6f;  alfa' Sigma_MV^-1 alfa = %.6f\n", sa - sf, g$a2_mv))
}
cat(sprintf("\nSR mensual de cada factor por separado: Mkt %.4f, SMB %.4f, HML %.4f, Mom %.4f\n",
            mean(F[, 1]) / sd(F[, 1]), mean(F[, 2]) / sd(F[, 2]), mean(F[, 3]) / sd(F[, 3]), mean(F[, 4]) / sd(F[, 4])))
