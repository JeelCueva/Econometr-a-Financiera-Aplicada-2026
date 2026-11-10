# Capítulo 5, Ejercicio 5.4: una regresión espuria con datos reales (oro y S&P 500)
source("codigo/R/cap05_funciones.R")
s <- read.csv("datos/internacional/sp500_mensual_shiller.csv"); s$mes <- substr(s$Date, 1, 7)
g <- read.csv("datos/internacional/oro_mensual.csv"); names(g) <- c("mes", "oro")
d <- merge(s[, c("mes", "SP500")], g, by = "mes")
d <- d[d$mes >= "1975-01" & d$mes <= "2026-09", ]
lo <- log(d$oro); ls <- log(d$SP500)
cat(sprintf("T = %d meses (%s a %s)\n", nrow(d), d$mes[1], d$mes[nrow(d)]))
for (n in c("lo", "ls")) {
  x <- get(n); cat(sprintf("ADF %s: niveles t = %.3f | diferencias t = %.3f\n", n, adf(x)$t, adf(diff(x))$t))
}
m <- lm(lo ~ ls); sm <- summary(m); e <- resid(m)
dw <- sum(diff(e)^2) / sum(e^2)
cat(sprintf("\nNiveles:     b = %.4f  t = %.2f  R2 = %.4f  DW = %.4f\n", coef(m)[2], sm$coefficients[2, 3], sm$r.squared, dw))
cat(sprintf("ADF de los residuos (sin constante): t = %.3f\n", adf(e, "n")$t))
md <- summary(lm(diff(lo) ~ diff(ls)))
cat(sprintf("Diferencias: b = %.4f  t = %.2f  R2 = %.5f\n", md$coefficients[2, 1], md$coefficients[2, 3], md$r.squared))
