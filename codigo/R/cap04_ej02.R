# Capítulo 4, Ejercicio 4.2: alfas y betas de las 12 industrias con errores robustos
d <- read.csv("datos/internacional/french_mensual_1949_2017.csv")
industrias <- c("NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
                "Telcm", "Utils", "Shops", "Hlth", "Money", "Other")
T <- nrow(d); X <- cbind(1, d$MktRF); XtXi <- solve(t(X) %*% X)
L <- floor(4 * (T / 100)^(2 / 9))            # regla de Newey y West (1994)
cat("Rezagos de Newey-West: L =", L, "\n\n")

hac <- function(e, L) {                      # matriz de Newey-West (L = 0: HC0)
  u <- X * e; S <- t(u) %*% u
  if (L > 0) for (j in 1:L) {
    G <- t(u[(j + 1):T, ]) %*% u[1:(T - j), ]
    S <- S + (1 - j / (L + 1)) * (G + t(G))
  }
  XtXi %*% S %*% XtXi
}
tabla <- t(sapply(industrias, function(i) {
  y <- d[[i]] - d$RF; b <- XtXi %*% t(X) %*% y; e <- as.vector(y - X %*% b)
  ee_cl <- sqrt(diag(sum(e^2) / (T - 2) * XtXi))
  ee_hc <- sqrt(diag(hac(e, 0) * T / (T - 2)))
  ee_nw <- sqrt(diag(hac(e, L)))
  c(t_alfa_cl = b[1] / ee_cl[1], t_alfa_HC1 = b[1] / ee_hc[1], t_alfa_NW = b[1] / ee_nw[1],
    ee_beta_cl = ee_cl[2], ee_beta_HC1 = ee_hc[2], ee_beta_NW = ee_nw[2])
}))
print(round(tabla, 4))
cat("\nAlfas significativas al 5 %: clásico", sum(abs(tabla[, 1]) > 1.96),
    "| HC1", sum(abs(tabla[, 2]) > 1.96), "| Newey-West", sum(abs(tabla[, 3]) > 1.96), "\n")
cat("Cociente medio ee(beta) NW/clásico:", round(mean(tabla[, 6] / tabla[, 4]), 3), "\n")
