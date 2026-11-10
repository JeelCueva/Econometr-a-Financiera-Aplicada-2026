# Capítulo 3, Ejercicio 3.5: el portafolio tangente de las 12 industrias y la crítica de Roll
d <- read.csv("datos/internacional/french_mensual_1949_2017.csv")
industrias <- c("NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
                "Telcm", "Utils", "Shops", "Hlth", "Money", "Other")
Z  <- as.matrix(d[, industrias]) - d$RF     # excesos de rendimiento
mu <- colMeans(Z); S <- cov(Z)
w  <- solve(S, mu); w <- w / sum(w)         # pesos del portafolio tangente
zq <- Z %*% w                               # exceso del portafolio tangente
cat("Pesos del portafolio tangente (%):\n"); print(round(100 * w, 1))
sharpe <- function(z) sqrt(12) * mean(z) / sd(z)
cat(sprintf("\nSharpe anual: tangente %.3f | mercado %.3f\n", sharpe(zq), sharpe(d$MktRF)))

# Betas respecto del tangente: la SML se cumple exactamente
b_q <- apply(Z, 2, function(z) cov(z, zq) / var(zq))
cat(sprintf("Máximo |mu_i - beta_iq * mu_q| = %.2e\n", max(abs(mu - b_q * mean(zq)))))
# Betas respecto del mercado: la SML no se cumple
b_m <- apply(Z, 2, function(z) cov(z, d$MktRF) / var(d$MktRF))
alfa_m <- mu - b_m * mean(d$MktRF)
cat(sprintf("Máximo |alfa_i| respecto del mercado = %.2f %% anual\n", 1200 * max(abs(alfa_m))))
