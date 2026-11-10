# Capítulo 3, Ejercicio 3.4: betas móviles con ventanas de 60 meses
d <- read.csv("datos/internacional/french_mensual_1949_2017.csv")
ventana <- 60
beta_movil <- function(i) {
  y <- d[[i]] - d$RF; x <- d$MktRF
  sapply(ventana:nrow(d), function(t) {
    k <- (t - ventana + 1):t
    cov(y[k], x[k]) / var(x[k])             # beta MCO en la ventana
  })
}
fin <- d$dates[ventana:nrow(d)]
for (i in c("Utils", "Enrgy", "BusEq")) {
  b <- beta_movil(i)
  cat(sprintf("%-6s min %.2f (%s)  max %.2f (%s)  último %.2f (%s)  desv %.2f\n", i,
              min(b), fin[which.min(b)], max(b), fin[which.max(b)], tail(b, 1), tail(fin, 1), sd(b)))
}
