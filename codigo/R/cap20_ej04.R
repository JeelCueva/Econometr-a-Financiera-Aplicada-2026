# Capítulo 20, Ejercicio 20.4: clasificación: la pendiente de la curva y las recesiones (Estrella y Mishkin, 1998)
source("codigo/R/cap20_funciones.R")
d <- read.csv("datos/internacional/tasas_cupon_cero_eeuu_mk.csv"); f <- d$fecha; r <- read.csv("datos/internacional/recesiones_nber.csv")
rec <- integer(length(f)); for (i in seq_len(nrow(r))) rec[f > r$pico[i] & f <= r$valle[i]] <- 1L
s <- d$r120 - d$r3; n3 <- d$r3; N <- length(f)
cat(sprintf("Datos mensuales %s a %s; meses en recesión: %d de %d\n", f[1], f[N], sum(rec), N))
cat("\nhorizonte h   b0        b1 (pendiente)      ee(b1)    pseudo R2   AUC      n\n")
for (h in c(3, 6, 12, 18, 24)) { yv <- rec[-(1:h)]; x <- s[1:(N - h)]; X <- cbind(1, x); L <- logit(yv, X)
  p0 <- mean(yv); ll0 <- length(yv) * (p0 * log(p0) + (1 - p0) * log(1 - p0))
  cat(sprintf("   %3d     %8.4f   %8.4f          %.4f    %8.4f   %.4f   %d\n", h, L$b[1], L$b[2], sqrt(L$V[2, 2]), 1 - L$ll / ll0, auc(yv, X %*% L$b), length(yv))) }
h <- 12; yv <- rec[-(1:h)]; x <- s[1:(N - h)]; lv <- n3[1:(N - h)]; X1 <- cbind(1, x); X2 <- cbind(X1, lv)
L1 <- logit(yv, X1); L2 <- logit(yv, X2); b1 <- L1$b; b2 <- L2$b
cat(sprintf("\nh = 12: agregar el nivel de la tasa a 3 meses: b = (%.4f, %.4f, %.4f); LR = %.2f; p = %.4f; AUC = %.4f\n", b2[1], b2[2], b2[3], 2 * (L2$ll - L1$ll), pchisq(2 * (L2$ll - L1$ll), 1, lower.tail = FALSE), auc(yv, X2 %*% b2)))
pr <- as.vector(1 / (1 + exp(-X1 %*% b1)))
cat(sprintf("Pendiente que da Pr(recesión en 12 meses) = 50 %%: %.4f puntos; con pendiente 0: %.4f; con pendiente 2: %.4f\n", -b1[1] / b1[2], 1 / (1 + exp(-b1[1])), 1 / (1 + exp(-b1[1] - 2 * b1[2]))))
for (u in c(0.5, mean(yv))) { cl <- as.integer(pr > u); tp <- sum(cl == 1 & yv == 1); fp <- sum(cl == 1 & yv == 0); fn <- sum(cl == 0 & yv == 1); tn <- sum(cl == 0 & yv == 0)
  cat(sprintf("  umbral %.4f: verdaderos positivos = %d, falsos positivos = %d, falsos negativos = %d, verdaderos negativos = %d; sensibilidad = %.4f; especificidad = %.4f\n", u, tp, fp, fn, tn, tp / (tp + fn), tn / (tn + fp))) }
ff <- f[-(1:h)]; i0 <- which(ff == "1965-01")
po <- sapply(i0:length(yv), function(i) { bb <- logit(yv[1:(i - h)], X1[1:(i - h), ])$b; 1 / (1 + exp(-sum(X1[i, ] * bb))) })
cat(sprintf("\nFuera de muestra (%s a %s, solo con recesiones ya observadas): AUC = %.4f; Pr media en meses de recesión = %.4f; en expansión = %.4f\n", ff[i0], ff[length(ff)], auc(yv[i0:length(yv)], po), mean(po[yv[i0:length(yv)] == 1]), mean(po[yv[i0:length(yv)] == 0])))
