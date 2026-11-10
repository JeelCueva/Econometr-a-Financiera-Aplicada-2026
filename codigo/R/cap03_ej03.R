# Capítulo 3, Ejercicio 3.3: la recta del mercado de valores (SML) con las 12 industrias
d <- read.csv("datos/internacional/french_mensual_1949_2017.csv")
industrias <- c("NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
                "Telcm", "Utils", "Shops", "Hlth", "Money", "Other")
beta  <- sapply(industrias, function(i) coef(lm(I(d[[i]] - d$RF) ~ d$MktRF))[2])
media <- sapply(industrias, function(i) 1200 * mean(d[[i]] - d$RF))  # % anual
prima <- 1200 * mean(d$MktRF)
cat(sprintf("Prima de mercado promedio: %.2f %% anual\n\n", prima))

sml <- lm(media ~ beta)                     # regresión de corte transversal
print(round(summary(sml)$coefficients, 3))
cat(sprintf("\nR2 de corte transversal = %.3f\n", summary(sml)$r.squared))
cat(sprintf("Pendiente estimada %.2f frente a la teórica %.2f; intercepto %.2f frente a 0\n",
            coef(sml)[2], prima, coef(sml)[1]))
