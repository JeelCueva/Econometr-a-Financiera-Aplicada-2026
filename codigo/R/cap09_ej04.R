# Capítulo 9, Ejercicio 9.4: ¿tiene el petróleo efecto apalancamiento? GJR y EGARCH para el Brent
source("codigo/R/cap09_funciones.R")
b <- read.csv("datos/internacional/brent_diario.csv"); r <- 100 * diff(log(b$Price)); T <- length(r)
m <- list(GARCH = ajustar(r, "garch", "c", "t"), GJR = ajustar(r, "gjr", "c", "t"), EGARCH = ajustar(r, "egarch", "c", "t"))
for (nm in c("GJR", "EGARCH")) {
  cat(sprintf("%s(1,1)-t:\n", nm))
  print(round(cbind(coef = m[[nm]]$p, ee_rob = m[[nm]]$ee_rob, t_rob = round(m[[nm]]$p / m[[nm]]$ee_rob, 2)), 4))
}
LR <- 2 * (m$GJR$ll - m$GARCH$ll)
cat(sprintf("\nLR (gamma = 0 en el GJR) = %.2f, p = %.4f\n", LR, pchisq(LR, 1, lower.tail = FALSE)))
cat("Modelo     k    log-ver        AIC        BIC\n")
for (nm in names(m)) cat(sprintf("%-8s %3d %10.2f %10.2f %10.2f\n", nm, m[[nm]]$k, m[[nm]]$ll,
                                 -2 * m[[nm]]$ll + 2 * m[[nm]]$k, -2 * m[[nm]]$ll + m[[nm]]$k * log(T)))
p <- m$GJR$p
cat(sprintf("GJR: coeficiente de e^2 si e < 0: %.4f; si e > 0: %.4f; cociente = %.2f\n",
            p["alpha"] + p["gamma"], p["alpha"], (p["alpha"] + p["gamma"]) / p["alpha"]))
s2 <- var(r); x <- 2 * sqrt(s2); q <- m$EGARCH$p
eg <- function(x) exp(q["omega"] + q["alpha"] * (abs(x / sqrt(s2)) - RAIZ_2_PI) + q["gamma"] * x / sqrt(s2) + q["beta"] * log(s2))
gj <- function(x) p["omega"] + (p["alpha"] + p["gamma"] * (x < 0)) * x^2 + p["beta"] * s2
cat(sprintf("choque de +-2 desviaciones (%.3f %%): cociente sigma2(-)/sigma2(+): GJR = %.2f; EGARCH = %.2f\n",
            x, gj(-x) / gj(x), eg(-x) / eg(x)))
sub <- b$Date[-1] <= "2019-12-31"
mg <- ajustar(r[sub], "gjr", "c", "t")
cat(sprintf("\nGJR-t sin 2020-2026 (hasta 2019, T = %d): gamma = %.4f (ee rob. %.4f, t = %.2f)\n",
            sum(sub), mg$p["gamma"], mg$ee_rob["gamma"], mg$p["gamma"] / mg$ee_rob["gamma"]))
