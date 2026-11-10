# Capítulo 4, Ejercicio 4.4: ¿predice la razón dividendo/precio el rendimiento a 12 meses?
s <- read.csv("datos/internacional/sp500_mensual_shiller.csv")
s <- s[as.Date(s$Date) <= as.Date("2023-06-01"), ]     # último mes con dividendos
P <- s$SP500; D <- s$Dividend; n <- length(P); k <- 12
g  <- log((P[-1] + D[-1] / 12) / P[-n])                 # rendimiento total mensual (log)
dp <- log(D / P)                                         # log de la razón dividendo/precio
y  <- sapply(1:(n - k), function(t) sum(g[t:(t + k - 1)]))   # rendimiento de t a t+12
x  <- dp[1:(n - k)]; T <- length(y)
X <- cbind(1, x); XtXi <- solve(t(X) %*% X)
b <- XtXi %*% t(X) %*% y; e <- as.vector(y - X %*% b)
nw <- function(L) {
  u <- X * e; S <- t(u) %*% u
  for (j in 1:L) { G <- t(u[(j + 1):T, ]) %*% u[1:(T - j), ]; S <- S + (1 - j / (L + 1)) * (G + t(G)) }
  sqrt(diag(XtXi %*% S %*% XtXi))
}
ee <- rbind(clasico = sqrt(diag(sum(e^2) / (T - 2) * XtXi)), NW_12 = nw(12), NW_24 = nw(24))
cat(sprintf("Muestra: %s a %s (T = %d regresiones traslapadas)\n", s$Date[1], s$Date[T], T))
cat(sprintf("b = %.4f   R2 = %.4f\n", b[2], summary(lm(y ~ x))$r.squared))
print(round(cbind(ee_b = ee[, 2], t_b = b[2] / ee[, 2]), 3))
enero <- format(as.Date(s$Date[1:T]), "%m") == "01"      # sin traslape: un dato por año
m <- summary(lm(y[enero] ~ x[enero]))$coefficients
cat(sprintf("Sin traslape (eneros, T = %d): b = %.4f  ee = %.4f  t = %.2f\n", sum(enero), m[2, 1], m[2, 2], m[2, 3]))
cat(sprintf("Autocorrelación de dp: rezago 1 = %.3f, rezago 12 = %.3f\n",
            cor(dp[-1], dp[-n]), cor(dp[-(1:12)], dp[1:(n - 12)])))
