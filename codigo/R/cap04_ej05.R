# Capítulo 4, Ejercicio 4.5: ¿cambió la beta de Utils? Chow, sup-F y CUSUM
d <- read.csv("datos/internacional/french_mensual_1949_2017.csv")
y <- d$Utils - d$RF; X <- cbind(1, d$MktRF); T <- length(y); k <- 2
src <- function(i) { m <- lm.fit(X[i, , drop = FALSE], y[i]); sum(m$residuals^2) }
chow <- function(tb) {                      # tb = primera observación del segundo régimen
  S <- src(1:T); S1 <- src(1:(tb - 1)); S2 <- src(tb:T)
  ((S - S1 - S2) / k) / ((S1 + S2) / (T - 2 * k))
}
tb <- which(d$dates == "1992-10")
F  <- chow(tb)
cat(sprintf("Chow con quiebre en 1992-10: F = %.2f  p = %.2e\n", F, pf(F, k, T - 2 * k, lower.tail = FALSE)))
b1 <- coef(lm(y[1:(tb - 1)] ~ X[1:(tb - 1), 2])); b2 <- coef(lm(y[tb:T] ~ X[tb:T, 2]))
cat(sprintf("Beta antes = %.3f  después = %.3f | alfa anual antes = %.2f %%  después = %.2f %%\n",
            b1[2], b2[2], 1200 * b1[1], 1200 * b2[1]))

# Versión robusta (Wald con HC1) mediante variables de interacción
Dq <- as.numeric(seq_len(T) >= tb)
Z <- cbind(X, Dq, Dq * X[, 2]); ZtZi <- solve(t(Z) %*% Z)
bz <- ZtZi %*% t(Z) %*% y; ez <- as.vector(y - Z %*% bz)
V  <- ZtZi %*% (t(Z * ez) %*% (Z * ez)) %*% ZtZi * T / (T - 4)
W  <- t(bz[3:4]) %*% solve(V[3:4, 3:4]) %*% bz[3:4]
cat(sprintf("Wald robusto (HC1): W = %.2f  p = %.2e\n", W, pchisq(W, 2, lower.tail = FALSE)))

# sup-F: fecha desconocida, recorte del 15 %
rango <- floor(0.15 * T):ceiling(0.85 * T)
Fs <- sapply(rango, chow)
cat(sprintf("sup-F = %.2f en %s (valor crítico asintótico al 5 %%: 11,7 / 2 = 5,85)\n",
            max(Fs), d$dates[rango[which.max(Fs)]]))

# CUSUM con residuos recursivos (Brown, Durbin y Evans, 1975)
w <- sapply((k + 1):T, function(t) {
  Xa <- X[1:(t - 1), ]; ba <- solve(t(Xa) %*% Xa, t(Xa) %*% y[1:(t - 1)])
  f  <- 1 + t(X[t, ]) %*% solve(t(Xa) %*% Xa) %*% X[t, ]
  (y[t] - X[t, ] %*% ba) / sqrt(f)
})
W_t <- cumsum(w) / sd(w)
banda <- 0.948 * (sqrt(T - k) + 2 * (seq_along(w)) / sqrt(T - k))
cruce <- which(abs(W_t) > banda)
cat(sprintf("CUSUM: máximo |W_t| / banda = %.2f; %s\n", max(abs(W_t) / banda),
            if (length(cruce) == 0) "no cruza las bandas" else paste("cruza en", d$dates[k + cruce[1]])))
