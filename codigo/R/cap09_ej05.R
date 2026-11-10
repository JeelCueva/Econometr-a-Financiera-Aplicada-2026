# Capítulo 9, Ejercicio 9.5: GARCH en media: ¿pagan más las acciones cuando la volatilidad es mayor?
source("codigo/R/cap09_funciones.R")
f <- read.csv("datos/internacional/french_mercado_mensual_1926_2018.csv"); x <- f$Mkt.RF; T <- length(x)
cat(sprintf("Exceso de rendimiento mensual del mercado (%%), %s a %s, T = %d\n", f$dates[1], f$dates[T], T))
cat(sprintf("media = %.4f; varianza = %.4f\n\n", mean(x), var(x)))
casos <- list("GARCH-M normal" = c("garch", "normal"), "GARCH-M t" = c("garch", "t"), "GJR-M t" = c("gjr", "t"))
for (nm in names(casos)) {
  m <- ajustar(x, casos[[nm]][1], "m", casos[[nm]][2]); p <- m$p
  pr <- p["lambda"] * mean(m$s2)
  cat(sprintf("%-15s lambda = %.5f (ee rob. %.5f, t = %.2f); mu = %.4f; log-ver = %.2f\n",
              nm, p["lambda"], m$ee_rob["lambda"], p["lambda"] / m$ee_rob["lambda"], p["mu"], m$ll))
  cat(sprintf("%-15s prima por volatilidad promedio: lambda x media(sigma2) = %.4f %% mensual (%.2f %% anual)\n",
              "", pr, 12 * pr))
}
