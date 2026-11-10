# Capítulo 2, Ejercicio 2.1: momentos y normalidad de los rendimientos diarios del Brent
datos <- read.csv("datos/internacional/brent_diario.csv")
datos$Date <- as.Date(datos$Date)
r      <- diff(log(datos$Price))          # rendimientos logarítmicos diarios
fechas <- datos$Date[-1]

momentos <- function(x) {                  # momentos muestrales con divisor T
  T <- length(x); d <- x - mean(x)
  m2 <- mean(d^2); s <- mean(d^3) / m2^1.5; k <- mean(d^4) / m2^2
  jb <- T / 6 * (s^2 + (k - 3)^2 / 4)      # estadístico de Jarque-Bera
  c(T = T, media = mean(x), desv_est = sqrt(m2), asimetria = s,
    curtosis = k, JB = jb, p_valor = pchisq(jb, df = 2, lower.tail = FALSE))
}
mostrar <- function(m) cat(sprintf("%-10s %12.5g\n", names(m), m), sep = "")
cat("Período:", format(min(fechas)), "a", format(max(fechas)), "\n\n")
mostrar(momentos(r))

cat("\nLos cinco rendimientos más grandes en valor absoluto:\n")
top <- order(abs(r), decreasing = TRUE)[1:5]
print(data.frame(fecha = fechas[top], r = round(r[top], 4)), row.names = FALSE)

cat("\nSin el 21 y el 22 de abril de 2020:\n")
sin_abr20 <- !(fechas %in% as.Date(c("2020-04-21", "2020-04-22")))
mostrar(momentos(r[sin_abr20]))

z <- (r - mean(r)) / sd(r)                 # rendimientos estandarizados
cat("\nP(|z| > 4): observada =", signif(mean(abs(z) > 4), 4),
    "| normal =", signif(2 * pnorm(-4), 4), "\n")
