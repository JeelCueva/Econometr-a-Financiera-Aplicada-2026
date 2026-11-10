# Capítulo 16, Ejercicio 16.1: Facebook y su anuncio de resultados del 25 de julio de 2018
source("codigo/R/cap16_funciones.R")
D <- cargar(); f <- D$fecha; R <- D$R; fb <- R[, 3]; m <- R[, 5]
t0 <- which(f == "2018-07-26"); cat(sprintf("día 0 = %s (posición %d); ventana de estimación: %s a %s\n", f[t0], t0 - 1, f[t0 - 250], f[t0 - 11]))
cat(sprintf("rendimiento de FB el día 0 = %.4f %%; del S&P 500 = %.4f %%\n\n", fb[t0], m[t0]))
cat("modelo      alfa      beta      s     AR(0)    ee(AR0)   t(AR0)   CAR(-1,+1)  ee     CAR(0,+5)   ee     CAR(-5,+5)   ee\n")
res <- list()
for (mod in c("mercado", "media", "ajustado")) {
  r <- rendimientos_normales(fb, m, t0, modelo = mod); res[[mod]] <- r; ee0 <- sqrt(r$V[6, 6])
  a <- if (mod == "ajustado") 0 else r$b[1]; b <- if (mod == "mercado") r$b[2] else if (mod == "ajustado") 1 else 0
  c1 <- car(r, -1, 1); c2 <- car(r, 0, 5); c3 <- car(r, -5, 5)
  cat(sprintf("%-9s %8.4f %8.4f %7.4f %9.4f %8.4f %8.2f  %9.4f %7.4f %9.4f %7.4f %9.4f %7.4f\n", mod, a, b, sqrt(r$s2), r$AR[6], ee0, r$AR[6] / ee0,
              c1[1], c1[2], c2[1], c2[2], c3[1], c3[2]))
}
r <- res$mercado
cat("\nModelo de mercado, ventana del evento:\n")
cat("  tau   fecha        R_FB      R_m       AR      ee(AR)     t      CAR\n")
for (k in seq_along(r$tau)) { tau <- r$tau[k]
  cat(sprintf("  %3d  %s  %8.4f %8.4f %8.4f %8.4f %7.2f %8.4f\n", tau, f[t0 + tau], fb[t0 + tau], m[t0 + tau], r$AR[k], sqrt(r$V[k, k]), r$AR[k] / sqrt(r$V[k, k]), sum(r$AR[1:k]))) }
cat(sprintf("  valor crítico t(%d) al 5 %% bilateral = %.4f\n", r$gl, qt(0.975, r$gl)))
ie <- c((t0 - 250):(t0 - 11), t0); dd <- as.numeric(ie == t0)
mm <- lm(fb[ie] ~ m[ie] + dd); cf <- summary(mm)$coef
cat(sprintf("\nRegresión con ficticia del día 0: delta = %.4f; ee = %.4f; t = %.2f\n", cf[3, 1], cf[3, 2], cf[3, 3]))
tc <- which(f == "2018-03-19"); rx <- rendimientos_normales(fb, m, t0, excluir = tc:(tc + 5))
cat(sprintf("Excluyendo %s a %s de la estimación: beta = %.4f; s = %.4f; AR(0) = %.4f; t = %.2f\n", f[tc], f[tc + 5], rx$b[2], sqrt(rx$s2), rx$AR[6], rx$AR[6] / sqrt(rx$V[6, 6])))
