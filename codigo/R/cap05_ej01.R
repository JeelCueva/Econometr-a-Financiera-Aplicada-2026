# Capítulo 5, Ejercicio 5.1: ¿tiene raíz unitaria el precio del Brent? ¿Y sus rendimientos?
source("codigo/R/cap05_funciones.R")
b <- read.csv("datos/internacional/brent_diario.csv")
lp <- log(b$Price); r <- diff(lp)
casos <- list("log precio (c)" = list(lp, "c"), "log precio (ct)" = list(lp, "ct"),
              "rendimiento (c)" = list(r, "c"))
for (n in names(casos)) {
  y <- casos[[n]][[1]]; det <- casos[[n]][[2]]
  a <- adf(y, det); k <- kpss(y, det)
  cat(sprintf("%-16s ADF: t = %8.3f (p = %2d rezagos; VC 5%% = %5.2f) | KPSS: eta = %6.3f (l = %d; VC 5%% = %.3f)\n",
              n, a$t, a$p, vc_adf[[det]]["5%"], k["eta"], k["l"], vc_kpss[[det]]["5%"]))
}
