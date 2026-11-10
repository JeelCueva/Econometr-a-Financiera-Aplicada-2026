# Capítulo 2, Ejercicio 2.5: índice de cola de Hill para pérdidas y ganancias del Brent
datos <- read.csv("datos/internacional/brent_diario.csv")
r <- diff(log(datos$Price))

hill <- function(x, k) {                   # x > 0; usa las k mayores observaciones
  xs <- sort(x[x > 0], decreasing = TRUE)
  k / sum(log(xs[1:k] / xs[k + 1]))
}
k <- c(50, 100, 200, 400)
tabla <- data.frame(k = k, pct_muestra = round(100 * k / length(r), 2),
                    alfa_perdidas = sapply(k, function(kk) hill(-r, kk)),
                    alfa_ganancias = sapply(k, function(kk) hill(r, kk)))
tabla$ee_perdidas  <- tabla$alfa_perdidas / sqrt(k)   # error estándar asintótico alfa/sqrt(k)
tabla$ee_ganancias <- tabla$alfa_ganancias / sqrt(k)
print(round(tabla, 3), row.names = FALSE)
