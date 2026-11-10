# Capítulo 5, Ejercicio 5.2: el VIX, una serie estacionaria pero persistente
source("codigo/R/cap05_funciones.R")
v <- read.csv("datos/internacional/vix_diario.csv")
y <- log(v$CLOSE); T <- length(y)
a <- adf(y, "c"); k <- kpss(y, "c")
cat(sprintf("T = %d días (%s a %s)\n", T, v$DATE[1], v$DATE[T]))
cat(sprintf("ADF: t = %.3f con %d rezagos (VC 5%% = -2,86)\n", a$t, a$p))
cat(sprintf("KPSS: eta = %.3f con l = %d (VC 5%% = 0,463)\n", k["eta"], k["l"]))
# AR(1) por MCO: y_t = c + phi * y_{t-1} + e_t
m <- lm(y[-1] ~ y[-T]); phi <- coef(m)[2]; mu <- coef(m)[1] / (1 - phi)
cat(sprintf("AR(1): phi = %.4f  media de largo plazo = %.3f (VIX = %.2f)\n", phi, mu, exp(mu)))
cat(sprintf("Vida media de un choque = ln(0,5)/ln(phi) = %.1f días\n", log(0.5) / log(phi)))
