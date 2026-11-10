# Capítulo 11, Ejercicio 11.3: el diferencial WTI-Brent y el cambio de régimen de 2011
source("codigo/R/cap11_funciones.R")
m <- crudo_mensual(); L <- log(as.matrix(m[, c("Price_b", "Price_w")])); s <- 100 * (L[, 2] - L[, 1])
per <- list("1987-05 a 2010-12" = m$mes <= "2010-12", "2011-01 a 2026-10" = m$mes >= "2011-01", "muestra completa" = rep(TRUE, nrow(m)))
cat("Diferencial s = 100 (log WTI - log Brent), en % del precio del Brent\n")
cat("Período               T    media  desv.est.  ADF s (p)        traza r=0  beta_WTI   LR beta=(1,-1)  valor p\n")
for (nm in names(per)) {
  k <- per[[nm]]; a <- adf(s[k], "c"); J <- johansen(L[k, ], 2); v <- vecm(L[k, ], 2); lr <- lr_beta(L[k, ], 2, matrix(c(1, -1)))
  cat(sprintf("%-18s %4d %8.2f %9.2f %7.2f (%d) %14.2f %10.4f %12.2f %12.4f\n", nm, sum(k), mean(s[k]), sd(s[k]),
              a$t, a$p, J$traza[1], v$beta[2], lr["LR"], pchisq(lr["LR"], 1, lower.tail = FALSE)))
}
cat(sprintf("\nmínimo del diferencial: %.2f %% (%s); máximo: %.2f %% (%s)\n", min(s), m$mes[which.min(s)], max(s), m$mes[which.max(s)]))
k <- m$mes >= "2011-01" & m$mes <= "2013-12"
cat(sprintf("promedio 2011-2013: %.2f %%; promedio 2014-2026: %.2f %%\n", mean(s[k]), mean(s[m$mes >= "2014-01"])))
