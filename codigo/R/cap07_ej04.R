# Capítulo 7, Ejercicio 7.4: prueba de rachas (Wald-Wolfowitz) sobre los signos
source("codigo/R/cap07_funciones.R")
b <- read.csv("datos/internacional/brent_diario.csv"); r <- diff(log(b$Price))
f <- read.csv("datos/internacional/french_mensual_1949_2017.csv")
series <- list("Brent diario" = r[r != 0], "S1V3 mensual" = f$S1V3[f$S1V3 != 0],
               "S5V3 mensual" = f$S5V3[f$S5V3 != 0])
cat(sprintf("Brent: %d días con rendimiento exactamente cero excluidos\n", sum(r == 0)))
cat("Serie            n1(+)  n2(-)  rachas  esperadas       z\n")
for (nm in names(series)) {
  x <- series[[nm]] > 0; k <- rachas(x)
  cat(sprintf("%-14s %6d %6d %7d %10.1f %7.2f\n", nm, k["n1"], k["n2"], k["R"], k["ER"], k["z"]))
}
