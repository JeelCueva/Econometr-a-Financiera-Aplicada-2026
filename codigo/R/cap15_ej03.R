# Capítulo 15, Ejercicio 15.3: rendimientos en exceso de mantener bonos y regresiones de Fama-Bliss
source("codigo/R/cap15_funciones.R")
D <- cargar(); Y <- D$Y; colnames(Y) <- paste0("r", TAU); T <- nrow(Y)
cat("Rendimiento en exceso a un mes del bono de n meses (en % mensual):\n")
cat("  rx_{t+1}(n) = (n/12) y_{n,t} - ((n-1)/12) y_{n-1,t+1} - y_{1,t}/12\n")
cat("Diferencial a plazo: (f_t(n) - y_{1,t})/12, con f_t(n) = n y_{n,t} - (n-1) y_{n-1,t}\n\n")
cat("   n   media rx   desv.est.   Sharpe anual  |   b (Fama-Bliss)   ee     t(b=0)    R2\n")
for (n in c(2, 3, 6, 12)) {
  yn <- Y[, paste0("r", n)]; yn1 <- Y[, paste0("r", n - 1)]; y1 <- Y[, "r1"]
  rx <- (n / 12) * yn[-T] - ((n - 1) / 12) * yn1[-1] - y1[-T] / 12
  fs <- (n * yn[-T] - (n - 1) * yn1[-T] - y1[-T]) / 12
  r <- mco_nw(rx, fs, 1)
  cat(sprintf("%4d %9.4f %10.4f %12.3f   | %12.4f %10.4f %8.2f %8.4f\n", n, mean(rx), sd(rx), sqrt(12) * mean(rx) / sd(rx), r$b[2], r$ee[2], r$b[2] / r$ee[2], r$r2))
}
