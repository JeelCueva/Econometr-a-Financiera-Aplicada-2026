# Capítulo 17, Ejercicio 17.2: el régimen volátil y las recesiones del NBER
source("codigo/R/cap17_funciones.R")
D <- mercado(); f <- D$fecha; y <- D$y; T <- length(y); m <- mean(y); sd0 <- sqrt(mean((y - m)^2))
r <- em_media_varianza(y, 2, c(m, m), c(0.7, 1.5) * sd0, matrix(c(0.95, 0.05, 0.05, 0.95), 2, 2))
p2 <- r$xs[, 2]; rec <- recesion_mensual(f); cl <- as.integer(p2 > 0.5)
cat(sprintf("meses en recesión (NBER): %d de %d\n", sum(rec), T))
cat(sprintf("Pr media del régimen volátil: en recesión = %.4f; en expansión = %.4f\n", mean(p2[rec == 1]), mean(p2[rec == 0])))
cat("\nTabla de clasificación (filas: NBER; columnas: régimen con Pr > 0,5)\n")
cat("              tranquilo   volátil\n")
cat(sprintf("  %s   %8d  %8d\n", c("expansión", "recesión "), c(sum(rec == 0 & cl == 0), sum(rec == 1 & cl == 0)), c(sum(rec == 0 & cl == 1), sum(rec == 1 & cl == 1))), sep = "")
cat(sprintf("Pr(volátil | recesión) = %.4f; Pr(volátil | expansión) = %.4f\n", mean(cl[rec == 1]), mean(cl[rec == 0])))
cat(sprintf("correlación entre Pr suavizada e indicador de recesión = %.4f\n", cor(p2, rec)))
cat("\nEpisodios del régimen volátil (Pr suavizada > 0,5) de 6 meses o más:\n")
cat("  inicio    fin      meses  rend. medio  desv. est.  meses en recesión\n")
ru <- rle(cl); fin <- cumsum(ru$lengths); ini <- fin - ru$lengths + 1
for (k in which(ru$values == 1 & ru$lengths >= 6)) { s <- y[ini[k]:fin[k]]
  cat(sprintf("  %s  %s  %5d  %10.4f  %10.4f  %8d\n", f[ini[k]], f[fin[k]], ru$lengths[k], mean(s), sqrt(mean((s - mean(s))^2)), sum(rec[ini[k]:fin[k]]))) }
cat("\nPr media del régimen volátil en cada recesión desde 1929:\n")
R <- read.csv("datos/internacional/recesiones_nber.csv")
for (i in seq_len(nrow(R))) { k <- f > R$pico[i] & f <= R$valle[i]
  if (sum(k) > 0 && R$pico[i] >= "1929") cat(sprintf("  %s a %s: %.4f  (%d meses)\n", R$pico[i], R$valle[i], mean(p2[k]), sum(k))) }
