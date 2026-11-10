# Capítulo 11, Ejercicio 11.1: ¿están cointegrados el Brent y el WTI? Prueba de Engle-Granger
source("codigo/R/cap11_funciones.R")
m <- crudo_mensual(); B <- log(m$Price_b); W <- log(m$Price_w); T <- length(B)
cat(sprintf("Promedios mensuales, %s a %s, T = %d meses\n\n", m$mes[1], m$mes[T], T))
cat("ADF con constante (rezagos por AIC); valor crítico al 5 %: -2.86\n")
for (et in c("log Brent", "log WTI")) {
  y <- if (et == "log Brent") B else W
  a <- adf(y, "c"); d <- adf(diff(y), "c")
  cat(sprintf("  %-10s nivel: t = %6.2f (p = %d)   diferencia: t = %7.2f (p = %d)\n", et, a$t, a$p, d$t, d$p))
}
cat("\nEngle-Granger:\n")
for (dir in c("Brent sobre WTI", "WTI sobre Brent")) {
  e <- if (dir == "Brent sobre WTI") engle_granger(B, W) else engle_granger(W, B)
  cat(sprintf("  %s: a = %.4f, b = %.4f; ADF de residuos t = %.3f (p = %d), gamma = %.4f\n",
              dir, e$a, e$b, e$t, e$p, e$gamma))
}
cat("  valores críticos (MacKinnon, 2010):", sprintf("%s %.3f", names(e$vc), e$vc), "\n")
e <- engle_granger(B, W); u <- e$u
cat(sprintf("\nresiduo: desv. est. = %.4f; mínimo %.4f (%s); máximo %.4f (%s)\n",
            sd(u), min(u), m$mes[which.min(u)], max(u), m$mes[which.max(u)]))
rho <- 1 + e$gamma; cat(sprintf("vida media aproximada del desequilibrio: ln 0.5 / ln(1 + gamma) = %.1f meses\n", log(0.5) / log(rho)))
