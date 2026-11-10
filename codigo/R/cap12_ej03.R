# Capítulo 12, Ejercicio 12.3: DCC trivariado (S&P 500, Nasdaq y Brent) y correlaciones en la crisis de 2008
source("codigo/R/cap12_funciones.R")
P <- panel_diario(); f <- P$fecha; e1 <- etapa1(P$R); m <- dcc_fit(e1$Z)
cat(sprintf("DCC(1,1) trivariado: a = %.4f (ee %.4f), b = %.4f (ee %.4f); a + b = %.4f\n", m$a, m$ee[1], m$b, m$ee[2], m$a + m$b))
cat(sprintf("LR frente a CCC = %.2f (2 g.l.)\n\n", 2 * (m$ll - m$ll_ccc)))
per <- list("calma 2006-01 a 2008-08" = f >= "2006-01-01" & f <= "2008-08-31",
            "crisis 2008-09 a 2009-03" = f >= "2008-09-01" & f <= "2009-03-31",
            "2017-2018" = f >= "2017-01-01")
pares <- list("S&P-Nasdaq" = c(1, 2), "S&P-Brent" = c(1, 3), "Nasdaq-Brent" = c(2, 3))
cat(sprintf("%-26s %12s %12s %12s\n", "promedio de la correlación DCC", names(pares)[1], names(pares)[2], names(pares)[3]))
for (nm in names(per)) cat(sprintf("%-26s %12.3f %12.3f %12.3f\n", nm,
  mean(m$R[per[[nm]], 1, 2]), mean(m$R[per[[nm]], 1, 3]), mean(m$R[per[[nm]], 2, 3])))
k <- per[[2]]
cat(sprintf("\nvolatilidad anual promedio (GARCH) en la crisis: S&P %.1f %%, Nasdaq %.1f %%, Brent %.1f %%\n",
            mean(sqrt(252 * e1$S2[k, 1])), mean(sqrt(252 * e1$S2[k, 2])), mean(sqrt(252 * e1$S2[k, 3]))))
mineig <- sapply(1:nrow(e1$Z), function(t) min(eigen(m$R[t, , ], symmetric = TRUE, only.values = TRUE)$values))
cat(sprintf("menor autovalor de R_t en toda la muestra: %.4f (todas las R_t son definidas positivas)\n", min(mineig)))
