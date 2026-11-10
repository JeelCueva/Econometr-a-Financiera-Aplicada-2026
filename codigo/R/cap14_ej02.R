# Capítulo 14, Ejercicio 14.2: la prueba GRS con tres conjuntos de carteras y tres modelos
source("codigo/R/cap14_funciones.R")
D <- datos()
conj <- list("tamaño y valor (9)" = VALOR, "tamaño y momentum (9)" = MOMENTUM, "industrias (12)" = INDUSTRIAS)
mods <- list(CAPM = "MktRF", FF3 = c("MktRF", "SMB", "HML"), Carhart = c("MktRF", "SMB", "HML", "Mom"))
cat("Carteras                 Modelo     GRS       gl         valor p     |alfa| medio   SR2 factores\n")
for (cn in names(conj)) for (mn in names(mods)) {
  g <- grs(D$R[, conj[[cn]]], D$F[, mods[[mn]], drop = FALSE])
  cat(sprintf("%-24s %-8s %7.3f  (%2d, %3d)  %10.2e   %8.3f      %8.4f\n", cn, mn, g$F, g$gl[1], g$gl[2], g$p, g$alfa_abs, g$sr2_f))
}
