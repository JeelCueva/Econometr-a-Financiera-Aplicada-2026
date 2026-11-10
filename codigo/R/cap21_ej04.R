# Capítulo 21, Etapa 4: VaR y ES de la cartera 1/N y su prueba retrospectiva (2016-2018)
source("codigo/R/cap21_funciones.R")
D <- cargar(); f <- D$fecha; R <- D$R; rp <- rowMeans(R[, 1:4]); n <- length(rp); ne <- sum(f < "2016-01-01"); L <- -rp
g <- garch_fit(rp[1:ne], "t"); p <- unname(g$par); s2 <- garch_s2(p, rp, g$bc); he <- ewma_var(rp)
PA <- list(); PP <- matrix(NA, n, 5); SS <- rep(NA, n)
for (anio in c("2016", "2017", "2018")) { i0 <- sum(f < paste0(anio, "-01-01")); i1 <- sum(f < paste0(as.integer(anio) + 1, "-01-01"))
  ga <- garch_fit(rp[1:i0], "t"); pa <- unname(ga$par); sa <- garch_s2(pa, rp[1:i1], ga$bc); PA[[anio]] <- pa
  for (t in (i0 + 1):i1) { PP[t, ] <- pa; SS[t] <- sa[t] } }
cat(sprintf("Cartera 1/N (rebalanceo diario); GARCH(1,1)-t estimado en 2014-2015: mu = %.4f, omega = %.4f, alfa = %.4f, beta = %.4f, nu = %.2f\n", p[1], p[2], p[3], p[4], p[5]))
for (a in c("2016", "2017", "2018")) cat(sprintf("  reestimado con datos hasta %d: alfa = %.4f, beta = %.4f, nu = %.2f\n", as.integer(a) - 1, PA[[a]][3], PA[[a]][4], PA[[a]][5]))
cat(sprintf("Prueba retrospectiva: %s a %s (%d días)\n\n", f[ne + 1], f[n], n - ne))
cat("nivel   excepciones  esperadas   Kupiec LR   p       Christoffersen LR   p      pérdida/ES en excepciones   método\n")
o <- (ne + 1):n
for (nivel in c(0.99, 0.975)) {
  M <- list("simulación histórica (250 días)" = t(sapply(o, function(t) var_es_hist(L[(t - 250):(t - 1)], nivel))),
            "EWMA normal" = t(sapply(o, function(t) var_es_normal(0, sqrt(he[t]), nivel))),
            "GARCH(1,1)-t" = t(sapply(o, function(t) var_es_t(p[1], sqrt(s2[t]), p[5], nivel))),
            "GARCH(1,1)-t reestimado cada año" = t(sapply(o, function(t) var_es_t(PP[t, 1], sqrt(SS[t]), PP[t, 5], nivel))))
  for (nom in names(M)) { VE <- M[[nom]]; x <- as.integer(L[o] > VE[, 1]); k <- sum(x); ku <- kupiec(x, nivel); ch <- christoffersen(x)
    ratio <- if (k > 0) mean(L[o][x == 1]) / mean(VE[x == 1, 2]) else NA
    cat(sprintf("%5.1f   %8d   %10.1f   %8.2f  %6.4f   %12.2f   %6.4f   %18.3f        %s\n", 100 * nivel, k, (1 - nivel) * (n - ne), ku[1], ku[2], ch[1], ch[2], ratio, nom)) } }
gf <- garch_fit(rp, "t"); q <- unname(gf$par); sf <- gf$s2; vf <- sqrt(q[2] + q[3] * (rp[n] - q[1])^2 + q[4] * sf[n]); VaRf <- var_es_t(q[1], vf, q[5], 0.99)
cat(sprintf("\nGARCH(1,1)-t con toda la muestra: alfa = %.4f, beta = %.4f, nu = %.2f\n", q[3], q[4], q[5]))
cat(sprintf("Pronóstico para el día siguiente a %s: volatilidad = %.4f %%; VaR 99 %% = %.4f %%; ES 99 %% = %.4f %%\n", f[n], vf, VaRf[1], VaRf[2]))
