# Apéndice A, Ejercicio A.1: error estándar de la razón media/desviación por el método delta
lee <- function(f) { x <- read.csv(file.path("datos/peru", f)); x$fecha <- as.Date(x$fecha); x }
series <- list(EPU = lee("epu_yahoo.csv")[, c("fecha", "cierre_ajustado")],
               SP500 = lee("gspc_yahoo.csv")[, c("fecha", "cierre")])
for (nom in names(series)) {
  x <- subset(series[[nom]], fecha <= as.Date("2025-12-31"))
  pm <- tapply(x[[2]], format(x$fecha, "%Y-%m"), function(v) tail(v, 1))
  R <- pm[-1] / pm[-length(pm)] - 1                       # rendimientos simples mensuales 2015-2025
  T <- length(R); mu <- mean(R); d <- R - mu
  s2 <- mean(d^2); s <- sqrt(s2); g3 <- mean(d^3) / s^3; g4 <- mean(d^4) / s2^2
  th <- mu / s
  v_norm <- 1 + th^2 / 2; v_gen <- 1 - th * g3 + th^2 * (g4 - 1) / 4
  cat(sprintf("%s: T = %d, media = %.5f, desv = %.5f, asimetría = %.3f, curtosis = %.3f\n",
              nom, T, mu, s, g3, g4))
  cat(sprintf("   theta mensual = %.4f;  anualizada x raiz(12) = %.4f\n", th, th * sqrt(12)))
  for (v in list(c("normal) ", v_norm), c("general)", v_gen))) {
    ee <- sqrt(as.numeric(v[2]) / T)
    cat(sprintf("   EE (%s = %.4f   IC 95%%: [%.4f, %.4f]\n", v[1], ee, th - 1.96 * ee, th + 1.96 * ee))
  }
}
