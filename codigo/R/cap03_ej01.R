# Capítulo 3, Ejercicio 3.1: beta del sector de servicios públicos (Utils), paso a paso
d <- read.csv("datos/internacional/french_mensual_1949_2017.csv")
y <- d$Utils - d$RF                         # exceso de rendimiento del sector
x <- d$MktRF                                # exceso de rendimiento del mercado
T <- length(y)

Sxx <- sum((x - mean(x))^2); Sxy <- sum((x - mean(x)) * (y - mean(y)))
beta  <- Sxy / Sxx
alfa  <- mean(y) - beta * mean(x)
e     <- y - alfa - beta * x                # residuos
s2    <- sum(e^2) / (T - 2)                 # varianza residual
ee_b  <- sqrt(s2 / Sxx)
ee_a  <- sqrt(s2 * (1 / T + mean(x)^2 / Sxx))
R2    <- 1 - sum(e^2) / sum((y - mean(y))^2)
cat(sprintf("T = %d meses (%s a %s)\n", T, d$dates[1], d$dates[T]))
cat(sprintf("beta = %.4f  ee = %.4f  t(beta=1) = %.2f\n", beta, ee_b, (beta - 1) / ee_b))
cat(sprintf("alfa = %.5f mensual (%.2f %% anual)  ee = %.5f  t(alfa=0) = %.2f\n",
            alfa, 1200 * alfa, ee_a, alfa / ee_a))
cat(sprintf("R2 = %.4f\n", R2))

# Descomposición del riesgo (varianzas anualizadas, divisor T-1)
v_tot <- 12 * var(y); v_sis <- 12 * beta^2 * var(x); v_idi <- 12 * var(e)
cat(sprintf("Varianza anual: total %.5f = sistemática %.5f + idiosincrática %.5f\n", v_tot, v_sis, v_idi))
cat(sprintf("Volatilidad anual: total %.2f %%, sistemática %.2f %%, idiosincrática %.2f %%\n",
            100 * sqrt(v_tot), 100 * sqrt(v_sis), 100 * sqrt(v_idi)))

cat("\nComprobación con lm():\n")
print(round(summary(lm(y ~ x))$coefficients, 5))
