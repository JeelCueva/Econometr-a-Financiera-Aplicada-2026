# Capítulo 3, Ejercicio 3.2: modelo de mercado para las 12 industrias
d <- read.csv("datos/internacional/french_mensual_1949_2017.csv")
industrias <- c("NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
                "Telcm", "Utils", "Shops", "Hlth", "Money", "Other")
tabla <- t(sapply(industrias, function(i) {
  m  <- summary(lm(I(d[[i]] - d$RF) ~ d$MktRF))
  cf <- m$coefficients
  c(alfa_anual_pct = 1200 * cf[1, 1], t_alfa = cf[1, 3],
    beta = cf[2, 1], ee_beta = cf[2, 2], t_beta1 = (cf[2, 1] - 1) / cf[2, 2],
    R2 = m$r.squared)
}))
print(round(tabla[order(tabla[, "beta"]), ], 3))
cat("\nIndustrias con |t(alfa)| > 1,96:", sum(abs(tabla[, "t_alfa"]) > 1.96), "de", nrow(tabla), "\n")
