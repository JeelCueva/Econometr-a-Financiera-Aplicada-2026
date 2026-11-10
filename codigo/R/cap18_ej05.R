# Capítulo 18, Ejercicio 18.5: ¿cambia la dependencia en las crisis? Crisis financiera de 2007-2009 frente al resto
source("codigo/R/cap18_funciones.R")
D <- indices(); f <- D$fecha; x <- D$x; y <- D$y; zx <- ewma_estandarizar(x); zy <- ewma_estandarizar(y)
crisis <- f >= "2007-07-01" & f <= "2009-06-30"
cat(sprintf("Crisis: %s a %s (%d días); resto: %d días\n", f[crisis][1], f[crisis][sum(crisis)], sum(crisis), sum(!crisis)))
M <- list("crisis" = crisis, "resto " = !crisis)
cat("\nmuestra   Pearson   tau      lambda_inf(5 %)  lambda_sup(5 %)\n")
for (nom in names(M)) { k <- M[[nom]]; l <- colas_empiricas(pseudo(zx[k]), pseudo(zy[k]), 0.05)
  cat(sprintf("%s   %.4f  %.4f   %10.4f      %10.4f\n", nom, cor(zx[k], zy[k]), cor(zx[k], zy[k], method = "kendall"), l[1], l[2])) }
cat("\nmuestra  cópula      parámetro(s)               log-verosim.  lambda_inf lambda_sup\n")
for (nom in names(M)) { k <- M[[nom]]; A <- ajustar(pseudo(zx[k]), pseudo(zy[k]))
  for (cc in c("gaussiana", "t", "clayton", "gumbel")) {
    if (cc == "t") { p <- A$t[1:2]; ll <- A$t[3]; ps <- sprintf("rho = %.4f, nu = %.3f", p[1], p[2]) }
    else { p <- A[[cc]][1]; ll <- A[[cc]][2]; ps <- sprintf("%s = %.4f", if (cc == "gaussiana") "rho" else "theta", p) }
    tl <- tau_lambda(cc, p); cat(sprintf("%s   %-10s  %-24s  %12.4f  %8.4f  %8.4f\n", nom, cc, ps, ll, tl[2], tl[3])) } }
ini <- seq(1, length(x) - 250 + 1, by = 21); tm <- sapply(ini, function(i) cor(zx[i:(i + 249)], zy[i:(i + 249)], method = "kendall"))
cat(sprintf("\ntau de Kendall móvil (ventanas de 250 días cada 21 días, %d ventanas): mínimo = %.4f (termina %s); máximo = %.4f (termina %s); mediana = %.4f\n",
            length(ini), min(tm), f[ini[which.min(tm)] + 249], max(tm), f[ini[which.max(tm)] + 249], median(tm)))
