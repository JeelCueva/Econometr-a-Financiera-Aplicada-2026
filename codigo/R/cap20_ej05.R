# Capítulo 20, Ejercicio 20.5: árboles de regresión y sobreajuste
source("codigo/R/cap20_funciones.R")
D <- cargar(); mes <- D$mes; X <- D$X[, IDX_MULT]; y <- D$y; corte <- which(mes == "1981-01"); n <- length(y)
ia <- 1:(corte - 1); ip <- corte:n; Xa <- X[ia, ]; ya <- y[ia]; Xp <- X[ip, ]; yp <- y[ip]
cat(sprintf("Entrenamiento: %s a %s (%d meses); prueba: %s a %s (%d meses); hoja mínima = 24 meses\n", mes[1], mes[corte - 1], length(ia), mes[corte], mes[n], length(ip)))
ref <- mean((yp - mean(ya))^2); Z <- cbind(1, Xa); b <- solve(crossprod(Z), crossprod(Z, ya))
ecm_l <- mean((yp - cbind(1, Xp) %*% b)^2); ecm_la <- mean((ya - Z %*% b)^2)
cat(sprintf("\nMCO lineal con 8 predictores: ECM entrenamiento = %.4f; ECM prueba = %.4f; R2 prueba = %.2f %%\n", ecm_la, ecm_l, 100 * (1 - ecm_l / ref)))
cat("\nprofundidad   hojas   ECM entrenamiento   ECM prueba   R2 prueba (%)\n")
for (d in 0:6) { t <- arbol(Xa, ya, d); ea <- mean((ya - predecir(t, Xa))^2); ep <- mean((yp - predecir(t, Xp))^2)
  cat(sprintf("   %5d      %4d     %12.4f     %10.4f    %8.2f\n", d, length(unique(predecir(t, Xa))), ea, ep, 100 * (1 - ep / ref))) }
cat("\nÁrbol de profundidad 2:\n"); imprimir_arbol(arbol(Xa, ya, 2), VARS_MULT)
