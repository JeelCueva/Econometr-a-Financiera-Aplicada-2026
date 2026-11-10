# Capítulo 16, Ejercicio 16.5: varianza inducida por el evento (simulación) y el VIX en los anuncios del FOMC
source("codigo/R/cap16_funciones.R")
D <- cargar(); R <- D$R; m <- R[, 5]; T <- length(m); F0 <- 251:T
E <- array(0, c(4, T, 240)); AR0 <- V0 <- matrix(NA, 4, T)
for (j in 1:4) for (t0 in F0) { x <- rendimientos_normales(R[, j], m, t0, evt = c(0, 0)); E[j, t0, ] <- x$e; AR0[j, t0] <- x$AR; V0[j, t0] <- x$V[1, 1] }
N <- 20; REP <- 1000; g <- lcg_nuevo(20162)
sorteo <- lapply(1:REP, function(r) t(sapply(1:N, function(i) {
  j <- floor(4 * lcg_u(g)) + 1; t0 <- F0[1] + floor(length(F0) * lcg_u(g)); u1 <- lcg_u(g); u2 <- lcg_u(g)
  c(j, t0, sqrt(-2 * log(u1)) * cos(2 * pi * u2)) })))                 # Box-Muller
cat("(a) Simulación: AR(0) + delta + ruido con varianza 1,25 V0 (varianza total x 2,25); rechazo si |est.| > 1,96\n")
cat("delta (%)    J1     Patell    BMP    signo   Corrado\n")
for (d in c(0, 1)) {
  rech <- numeric(5)
  for (s in sorteo) {
    jt <- s[, 1:2]; a <- AR0[jt] + d + s[, 3] * sqrt(1.25 * V0[jt]); p <- pruebas(a, V0[jt], rep(238, N))
    A <- cbind(t(sapply(1:N, function(i) E[s[i, 1], s[i, 2], ])), a)
    rech <- rech + (abs(c(p$J1, p$Patell, p$BMP, p$signo, corrado(A))) > 1.96)
  }
  cat(sprintf("%6.1f   %s\n", d, paste(sprintf("%7.3f", rech / REP), collapse = "  ")))
}
v <- read.csv("datos/internacional/vix_diario.csv"); fv <- v$DATE[-1]; x <- 100 * diff(log(v$CLOSE))
ev <- read.csv("datos/internacional/fomc_anuncios_2014_2018.csv")$fecha_anuncio
T0 <- match(ev, fv); res <- lapply(T0, function(t0) rendimientos_normales(x, x, t0, modelo = "media"))
AR <- t(sapply(res, `[[`, "AR")); V <- t(sapply(res, function(r) diag(r$V))); gl <- sapply(res, `[[`, "gl")
cat(sprintf("\n(b) Cambio porcentual del VIX, %d anuncios del FOMC; estimación [-250, -11]\n", length(T0)))
cat(" tau    CAAR      J1    Patell    BMP    signo  positivos  var(SAR)\n")
for (k in 3:9) { p <- pruebas(AR[, k], V[, k], gl)
  cat(sprintf(" %3d %8.4f %7.2f %7.2f %7.2f %7.2f   %2d/%d   %7.4f\n", k - 6, p$CAAR, p$J1, p$Patell, p$BMP, p$signo, p$pos, length(T0), var(p$SAR))) }
c <- rowSums(AR[, 5:6]); vv <- sapply(res, function(r) sum(r$V[5:6, 5:6])); sc <- c / sqrt(vv)
cat(sprintf("CAAR(-1,0) = %.4f; J1 = %.2f; BMP = %.2f\n", mean(c), mean(c) / sqrt(sum(vv) / length(c)^2), mean(sc) / (sd(sc) / sqrt(length(c)))))
