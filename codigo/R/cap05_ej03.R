# Capítulo 5, Ejercicio 5.3: la tasa de interés de largo plazo de EE. UU., ADF paso a paso
source("codigo/R/cap05_funciones.R")
s <- read.csv("datos/internacional/sp500_mensual_shiller.csv")
s <- s[as.Date(s$Date) <= as.Date("2023-06-01"), ]
i <- s$Long.Interest.Rate                   # rendimiento del bono del Tesoro a 10 años (%)
cat(sprintf("T = %d meses (%s a %s)\n", length(i), s$Date[1], s$Date[length(i)]))
a <- adf(i, "c")
cat(sprintf("Rezagos elegidos por AIC: p = %d\n", a$p))
cat("Regresión ADF (primeros coeficientes):\n")
nom <- c("gamma (y_{t-1})", paste0("delta_", 1:a$p), "constante")
print(round(data.frame(coef = a$coef, ee = a$ee, row.names = nom)[c(1, 2, 3, a$p + 2), ], 5))
cat(sprintf("t de gamma = %.5f / %.5f = %.3f  (VC: 1%% -3,43 | 5%% -2,86 | 10%% -2,57)\n", a$coef[1], a$ee[1], a$t))
ad <- adf(diff(i), "c")
cat(sprintf("En diferencias: t = %.3f con %d rezagos\n", ad$t, ad$p))
cat(sprintf("KPSS en niveles: eta = %.3f | en diferencias: eta = %.3f (VC 5%% = 0,463)\n",
            kpss(i)["eta"], kpss(diff(i))["eta"]))
