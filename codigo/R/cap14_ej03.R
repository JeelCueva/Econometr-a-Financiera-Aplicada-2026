# Capítulo 14, Ejercicio 14.3: primas por riesgo de Fama-MacBeth con 30 carteras
source("codigo/R/cap14_funciones.R")
D <- datos(); R <- D$R; T <- nrow(R)
cat(sprintf("30 carteras (9 tamaño-valor, 9 tamaño-momentum, 12 industrias), T = %d meses\n", T))
for (mn in c("FF3", "Carhart")) {
  fac <- if (mn == "FF3") c("MktRF", "SMB", "HML") else c("MktRF", "SMB", "HML", "Mom")
  fm <- fama_macbeth(R, D$F[, fac, drop = FALSE]); nm <- c("constante", fac)
  cat(sprintf("\n%s: R2 de corte transversal = %.3f; lambda' Omega^-1 lambda = %.4f\n", mn, fm$r2_ct, fm$c))
  cat("              lambda    ee FM   t FM   ee Shanken  t Shanken   media del factor\n")
  for (j in seq_along(nm)) cat(sprintf("%-10s %9.4f %8.4f %6.2f %10.4f %9.2f   %s\n", nm[j], fm$lam[j], fm$ee_fm[j], fm$lam[j] / fm$ee_fm[j],
                                       fm$ee_sh[j], fm$lam[j] / fm$ee_sh[j], if (j == 1) "" else sprintf("%.4f", mean(D$F[, fac[j - 1]]))))
}
