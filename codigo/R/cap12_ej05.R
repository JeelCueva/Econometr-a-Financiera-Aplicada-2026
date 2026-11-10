# Capítulo 12, Ejercicio 12.5: ¿contagio en 2008? La corrección de Forbes y Rigobon
source("codigo/R/cap12_funciones.R")
P <- panel_diario(); R <- P$R; f <- P$fecha
calma <- f >= "2006-01-01" & f <= "2008-08-31"; crisis <- f >= "2008-09-01" & f <= "2009-03-31"
nt <- sum(calma); nc <- sum(crisis); x <- R[, 1]
delta <- var(x[crisis]) / var(x[calma]) - 1
cat(sprintf("Calma: %d días; crisis: %d días. Varianza diaria del S&P 500: %.4f en calma y %.4f en crisis; delta = %.4f\n\n",
            nt, nc, var(x[calma]), var(x[crisis]), delta))
cat("Mercado       rho calma   rho crisis   rho crisis ajustada   z (sin ajuste)   z (ajustada)\n")
for (j in 2:3) {
  y <- R[, j]; rt <- cor(x[calma], y[calma]); rc <- cor(x[crisis], y[crisis])
  ra <- rc / sqrt(1 + delta * (1 - rc^2))
  ee <- sqrt(1 / (nc - 3) + 1 / (nt - 3))
  cat(sprintf("%-10s %11.4f %12.4f %20.4f %16.2f %14.2f\n", c("", "Nasdaq", "Brent")[j], rt, rc, ra,
              (atanh(rc) - atanh(rt)) / ee, (atanh(ra) - atanh(rt)) / ee))
}
