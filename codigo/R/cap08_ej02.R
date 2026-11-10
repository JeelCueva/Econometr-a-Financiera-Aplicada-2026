# Capítulo 8, Ejercicio 8.2: GARCH(1,1) normal para el Brent, estimado a mano por máxima verosimilitud
source("codigo/R/cap08_funciones.R")
b <- read.csv("datos/internacional/brent_diario.csv"); r <- 100 * diff(log(b$Price)); T <- length(r)
m <- garch_fit(r, "normal"); p <- m$par
tab <- cbind(coef = p, ee_hess = m$ee, t_hess = round(p / m$ee, 2), ee_rob = m$ee_rob, t_rob = round(p / m$ee_rob, 2))
cat(sprintf("T = %d; log-verosimilitud = %.4f; varianza inicial (backcast) = %.4f\n", T, m$ll, m$bc))
print(round(tab, 4))
pers <- p["alpha"] + p["beta"]; s2bar <- p["omega"] / (1 - pers)
cat(sprintf("persistencia alpha + beta = %.5f; vida media = %.1f días\n", pers, log(0.5) / log(pers)))
cat(sprintf("varianza incondicional = %.3f (muestral %.3f); volatilidad anual = %.1f %%\n",
            s2bar, var(r), sqrt(252 * s2bar)))
cat(sprintf("condición de cuarto momento: (a+b)^2 + 2a^2 = %.4f\n", pers^2 + 2 * p["alpha"]^2))
vol <- sqrt(252 * m$s2)[-(1:250)]; f <- b$Date[-(1:251)]   # se omite el primer año (arranque)
cat(sprintf("volatilidad anualizada desde %s: mínima %.1f %% (%s), máxima %.1f %% (%s)\n",
            f[1], min(vol), f[which.min(vol)], max(vol), f[which.max(vol)]))
