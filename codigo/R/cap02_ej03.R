# Capítulo 2, Ejercicio 2.3: gaussianidad agregativa (Brent)
datos <- read.csv("datos/internacional/brent_diario.csv")
r <- diff(log(datos$Price))

agregar <- function(x, h) {                # suma de bloques de h días, sin traslape
  n <- floor(length(x) / h)
  colSums(matrix(x[1:(n * h)], nrow = h))
}
resumen <- function(x) {
  d <- x - mean(x); m2 <- mean(d^2)
  s <- mean(d^3) / m2^1.5; k <- mean(d^4) / m2^2; T <- length(x)
  jb <- T / 6 * (s^2 + (k - 3)^2 / 4)
  c(n = T, desv_est = sqrt(m2), asimetria = s, curtosis = k,
    p_JB = pchisq(jb, 2, lower.tail = FALSE))
}
h <- c(1, 5, 21, 63, 126)
tabla <- t(sapply(h, function(hh) resumen(agregar(r, hh))))
rownames(tabla) <- paste0("h=", h)
print(round(tabla, 4))
cat("\nRaíz del tiempo: desv(h=21)/desv(h=1) =", round(tabla["h=21", "desv_est"] / tabla["h=1", "desv_est"], 3),
    " vs sqrt(21) =", round(sqrt(21), 3), "\n")
