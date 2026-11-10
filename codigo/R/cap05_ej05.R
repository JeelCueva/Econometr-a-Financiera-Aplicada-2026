# Capítulo 5, Ejercicio 5.5: ¿raíz unitaria o quiebre? Zivot-Andrews para la razón dividendo/precio
source("codigo/R/cap05_funciones.R")
s <- read.csv("datos/internacional/sp500_mensual_shiller.csv")
s <- s[as.Date(s$Date) <= as.Date("2023-06-01"), ]
y <- log(s$Dividend / s$SP500); T <- length(y)
a <- adf(y, "ct"); p <- a$p
cat(sprintf("ADF con tendencia: t = %.3f con %d rezagos (VC 5%% = -3,41)\n", a$t, p))
# Zivot-Andrews (cambio de nivel): dy_t = c + theta*DU_t + b*t + gamma*y_{t-1} + sum delta_j dy_{t-j}
dy <- diff(y); k <- (p + 1):length(dy)
Xb <- cbind(ylag = y[k], sapply(1:p, function(j) dy[k - j]), const = 1, tend = k)
corte <- floor(0.15 * T)
cands <- (corte + 2):(T - corte + 1)        # TB: primer período del nuevo régimen (índice de y)
tz <- sapply(cands, function(TB) {
  X <- cbind(Xb, DU = as.numeric(k + 1 >= TB))     # dy[k] corresponde al período k + 1
  m <- lm.fit(X, dy[k]); e <- m$residuals
  V <- sum(e^2) / (length(e) - ncol(X)) * chol2inv(qr.R(m$qr))
  m$coefficients[1] / sqrt(V[1, 1])
})
i <- which.min(tz)
cat(sprintf("Zivot-Andrews: t mínimo = %.3f en %s (VC al 5%%: -4,80)\n", tz[i], s$Date[cands[i]]))
