# Capítulo 9, Ejercicio 9.2: GJR-GARCH-t para el S&P 500 diario y comparación con el VIX
source("codigo/R/cap09_funciones.R")
s <- read.csv("datos/internacional/sp500_diario_yahoo.csv"); r <- 100 * diff(log(s$Adj.Close)); T <- length(r)
g <- ajustar(r, "garch", "c", "t"); j <- ajustar(r, "gjr", "c", "t"); p <- j$p
cat(sprintf("T = %d (%s a %s)\n\nGJR-GARCH(1,1)-t:\n", T, s$Date[2], s$Date[T + 1]))
print(round(cbind(coef = p, ee_rob = j$ee_rob, t_rob = round(p / j$ee_rob, 2)), 4))
cat(sprintf("\nGARCH(1,1)-t: alpha = %.4f, beta = %.4f, log-ver = %.2f\n", g$p["alpha"], g$p["beta"], g$ll))
cat(sprintf("GJR-GARCH-t: log-ver = %.2f; LR (gamma = 0) = %.2f, p = %.2e\n",
            j$ll, 2 * (j$ll - g$ll), pchisq(2 * (j$ll - g$ll), 1, lower.tail = FALSE)))
pers <- p["alpha"] + p["gamma"] / 2 + p["beta"]; s2bar <- p["omega"] / (1 - pers)
cat(sprintf("persistencia alpha + gamma/2 + beta = %.5f; vida media = %.1f días\n", pers, log(0.5) / log(pers)))
cat(sprintf("varianza incondicional = %.4f (muestral %.4f); volatilidad anual = %.1f %%\n",
            s2bar, var(r), sqrt(252 * s2bar)))
cat(sprintf("coeficiente de e_{t-1}^2: si e < 0, alpha + gamma = %.4f; si e > 0, alpha = %.4f\n",
            p["alpha"] + p["gamma"], p["alpha"]))
v <- read.csv("datos/internacional/vix_diario.csv")
d <- s$Date[-1]; vol <- sqrt(252 * j$s2); k <- match(d, v$DATE); ok <- !is.na(k)
cat(sprintf("\nVIX: %d días en común; correlación con la volatilidad GJR = %.3f\n",
            sum(ok), cor(vol[ok], v$CLOSE[k[ok]])))
cat(sprintf("promedios: VIX = %.2f; volatilidad GJR = %.2f; diferencia = %.2f puntos\n",
            mean(v$CLOSE[k[ok]]), mean(vol[ok]), mean(v$CLOSE[k[ok]] - vol[ok])))
