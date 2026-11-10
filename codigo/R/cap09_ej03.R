# Capítulo 9, Ejercicio 9.3: EGARCH-t para el S&P 500, comparación de modelos y curvas de impacto de noticias
source("codigo/R/cap09_funciones.R")
s <- read.csv("datos/internacional/sp500_diario_yahoo.csv"); r <- 100 * diff(log(s$Adj.Close)); T <- length(r)
m <- list(GARCH = ajustar(r, "garch", "c", "t"), GJR = ajustar(r, "gjr", "c", "t"), EGARCH = ajustar(r, "egarch", "c", "t"))
p <- m$EGARCH$p
cat("EGARCH(1,1)-t:\n")
print(round(cbind(coef = p, ee_rob = m$EGARCH$ee_rob, t_rob = round(p / m$EGARCH$ee_rob, 2)), 4))
cat(sprintf("pendiente de g(z): z > 0: alpha + gamma = %.4f; z < 0: gamma - alpha = %.4f\n",
            p["alpha"] + p["gamma"], p["gamma"] - p["alpha"]))
cat(sprintf("vida media (en logaritmos) = %.1f días\n\n", log(0.5) / log(p["beta"])))
cat("Modelo     k    log-ver        AIC        BIC\n")
for (nm in names(m)) cat(sprintf("%-8s %3d %10.2f %10.2f %10.2f\n", nm, m[[nm]]$k, m[[nm]]$ll,
                                 -2 * m[[nm]]$ll + 2 * m[[nm]]$k, -2 * m[[nm]]$ll + m[[nm]]$k * log(T)))
nic <- function(mod, x, s2) {                       # varianza de mañana como función del choque de hoy
  q <- m[[mod]]$p
  if (mod == "EGARCH") { z <- x / sqrt(s2)
    return(exp(q["omega"] + q["alpha"] * (abs(z) - RAIZ_2_PI) + q["gamma"] * z + q["beta"] * log(s2))) }
  g <- if (mod == "GJR") q["gamma"] else 0
  q["omega"] + (q["alpha"] + g * (x < 0)) * x^2 + q["beta"] * s2
}
s2 <- var(r); x <- 2 * sqrt(s2)
cat(sprintf("\nCurva de impacto con sigma2_{t-1} = varianza muestral (%.4f); choque de +-2 desviaciones (%.3f %%):\n", s2, x))
for (nm in names(m)) {
  a <- nic(nm, -x, s2); b <- nic(nm, x, s2)
  cat(sprintf("  %-7s sigma2(-) = %.4f  sigma2(+) = %.4f  cociente = %.2f\n", nm, a, b, a / b))
}
