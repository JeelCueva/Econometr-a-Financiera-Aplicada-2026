# Capítulo 1, Ejercicio 1.2: rendimientos simples y logarítmicos del índice BVL (BCRP) y del S&P 500 (Yahoo)
lee <- function(f) { x <- read.csv(f); x$fecha <- as.Date(x$fecha); x }
bvl <- lee("datos/peru/bvl_indice_general_bcrp.csv")
sp  <- lee("datos/peru/gspc_yahoo.csv")

cat("BVL alrededor del dato atípico:\n")
print(subset(bvl, fecha >= as.Date("2024-06-19") & fecha <= as.Date("2024-06-25")), row.names = FALSE)
bvl <- subset(bvl, fecha != as.Date("2024-06-21"))    # dato erróneo del 21-jun-2024
cat("Días con índice repetido (rendimiento cero):", sum(diff(bvl$indice) == 0), "\n\n")

analiza <- function(nom, fecha, p) {
  R <- p[-1] / p[-length(p)] - 1; r <- diff(log(p)); f <- fecha[-1]
  k <- which.max(R - r)
  cat(sprintf("%s: T = %d, r <= R en todas las fechas: %s\n", nom, length(r), all(r <= R + 1e-15)))
  cat(sprintf("   mayor diferencia R - r = %.5f el %s (R = %.4f, r = %.4f)\n",
              (R - r)[k], format(f[k]), R[k], r[k]))
  for (fr in c("%Y-%m", "%Y")) {
    g <- format(f, fr)
    suma <- tapply(r, g, sum); prod <- log(tapply(1 + R, g, prod))
    cat(sprintf("   %-8s: períodos = %d, máx |suma log - log prod| = %.1e\n",
                ifelse(fr == "%Y", "anual", "mensual"), length(suma), max(abs(suma - prod))))
  }
  list(r = r, anual = exp(tapply(r, format(f, "%Y"), sum)) - 1)
}
a <- analiza("BVL", bvl$fecha, bvl$indice)
b <- analiza("SP500", sp$fecha, sp$cierre)
cat("\nRendimientos simples anuales (%), 2026 hasta noviembre:\n")
print(round(100 * cbind(BVL = a$anual, SP500 = b$anual), 2))
est <- function(r) c(media = mean(r), desv_est = sd(r), minimo = min(r), maximo = max(r))
cat("\nRendimientos logarítmicos diarios:\n")
print(round(cbind(BVL = est(a$r), SP500 = est(b$r)), 5))
