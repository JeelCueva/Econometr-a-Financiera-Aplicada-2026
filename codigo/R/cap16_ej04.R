# Capítulo 16, Ejercicio 16.4: tamaño y potencia de las pruebas (simulación de Brown y Warner con datos reales)
source("codigo/R/cap16_funciones.R")
D <- cargar(); R <- D$R; m <- R[, 5]; T <- length(m); F0 <- 251:T          # días 0 posibles (índices de R)
E <- array(0, c(4, T, 240)); AR0 <- V0 <- matrix(NA, 4, T)
for (j in 1:4) for (t0 in F0) { x <- rendimientos_normales(R[, j], m, t0, evt = c(0, 0)); E[j, t0, ] <- x$e; AR0[j, t0] <- x$AR; V0[j, t0] <- x$V[1, 1] }
GL <- 238; N <- 20; REP <- 1000; DELTAS <- c(0, 0.5, 1, 2)
g <- lcg_nuevo(20161)
sorteo <- lapply(1:REP, function(r) t(sapply(1:N, function(i) { j <- floor(4 * lcg_u(g)) + 1; c(j, F0[1] + floor(length(F0) * lcg_u(g))) })))
cat(sprintf("%d fechas posibles x 4 acciones; %d muestras de N = %d eventos; rechazo si |estadístico| > 1,96\n\n", length(F0), REP, N))
cat("delta (%)    J1     Patell    BMP    signo   Corrado\n")
for (d in DELTAS) {
  rech <- numeric(5)
  for (s in sorteo) {
    a <- AR0[s] + d; v <- V0[s]; p <- pruebas(a, v, rep(GL, N))
    A <- cbind(t(sapply(1:N, function(i) E[s[i, 1], s[i, 2], ])), a)
    rech <- rech + (abs(c(p$J1, p$Patell, p$BMP, p$signo, corrado(A))) > 1.96)
  }
  cat(sprintf("%6.1f   %s\n", d, paste(sprintf("%7.3f", rech / REP), collapse = "  ")))
}
a <- as.vector(t(AR0[, F0])); sk <- mean((a - mean(a))^3) / (sqrt(mean((a - mean(a))^2)))^3
cat(sprintf("\nAR(0) de todas las fechas posibles: media = %.4f; mediana = %.4f; fracción positiva = %.4f; asimetría = %.4f\n", mean(a), median(a), mean(a > 0), sk))
