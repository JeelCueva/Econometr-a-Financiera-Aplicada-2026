# Capítulo 1, Ejercicio 1.4: portafolio 1/3 BAP, SCCO y BVN; rendimiento exacto frente a suma de logaritmos
lee <- function(t) { x <- read.csv(sprintf("datos/peru/%s_yahoo.csv", t)); x$fecha <- as.Date(x$fecha); x }
bap <- lee("bap"); scco <- lee("scco"); bvn <- lee("bvn")      # mismas 2981 fechas (NYSE)
k <- bap$cierre_ajustado / bap$cierre
i <- which(bap$fecha == as.Date("2020-04-08"))                 # ajuste duplicado de Yahoo en BAP
bap$cierre_ajustado[1:(i - 1)] <- bap$cierre_ajustado[1:(i - 1)] * k[i] / k[i - 1]
P <- cbind(BAP = bap$cierre_ajustado, SCCO = scco$cierre_ajustado, BVN = bvn$cierre_ajustado)

mes <- format(bap$fecha, "%Y-%m")
fin <- tapply(seq_along(mes), mes, max)                        # último día de cada mes
Pm  <- P[fin, ]
R <- Pm[-1, ] / Pm[-nrow(Pm), ] - 1; r <- diff(log(Pm)); w <- rep(1/3, 3)
Rp <- R %*% w; rp <- log1p(Rp); aprox <- r %*% w; err <- as.vector(rp - aprox)
disp <- apply(r, 1, function(x) mean((x - mean(x))^2))         # dispersión transversal
cat(sprintf("Meses: %d (%s a %s)\n", length(err), names(fin)[2], names(fin)[length(fin)]))
cat(sprintf("Error mínimo = %.2e (Prop. 1.14 exige >= 0): %s\n", min(err), all(err >= -1e-15)))
cat(sprintf("Error medio = %.2f pb, máximo = %.2f pb en %s\n", 1e4 * mean(err), 1e4 * max(err),
            names(fin)[-1][which.max(err)]))
cat(sprintf("Media de disp/2 = %.2f pb; correlación(error, disp/2) = %.4f\n", 1e4 * mean(disp / 2), cor(err, disp)))
b <- coef(lm(err ~ disp))
cat(sprintf("Regresión error = a + b*disp:  a = %.2e, b = %.4f  (teoría de segundo orden: 0,5)\n", b[1], b[2]))
cat(sprintf("\nSuma de errores en %d meses: %.2f puntos log\n", length(err), 100 * sum(err)))

crec <- Pm[nrow(Pm), ] / Pm[1, ]
cat(sprintf("\nValor final de 1 dólar (dic-2014 a %s):\n", names(fin)[length(fin)]))
cat(sprintf("   rebalanceo mensual: %.4f   comprar y mantener: %.4f\n", exp(sum(rp)), sum(w * crec)))
cat("   valor final por acción (comprar y mantener):\n"); print(round(crec, 4))
cat("   pesos finales sin rebalanceo:\n"); print(round(w * crec / sum(w * crec), 4))
