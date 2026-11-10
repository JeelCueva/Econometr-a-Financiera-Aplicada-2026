# Capítulo 21, Etapa 3: volatilidad: GARCH(1,1)-t estimado en 2014-2015 y evaluado fuera de muestra en 2016-2018
source("codigo/R/cap21_funciones.R")
D <- cargar(); f <- D$fecha; R <- D$R; ne <- sum(f < "2016-01-01"); n <- length(f)
cat(sprintf("Estimación: %s a %s (%d días); evaluación: %s a %s (%d días); parámetros fijos, varianza filtrada día a día\n", f[1], f[ne], ne, f[ne + 1], f[n], n - ne))
cat("\nserie    mu       omega     alfa     beta     nu     alfa+beta   vida media (días)\n")
H <- list()
for (j in seq_along(NOMBRES)) { g <- garch_fit(R[1:ne, j], "t"); p <- unname(g$par); s2 <- garch_s2(p, R[, j], g$bc); H[[NOMBRES[j]]] <- list(p = p, s2 = s2)
  cat(sprintf("%-6s  %6.3f  %8.4f  %7.4f  %7.4f  %6.2f   %8.4f   %10.1f\n", NOMBRES[j], p[1], p[2], p[3], p[4], p[5], p[3] + p[4], log(0.5) / log(p[3] + p[4]))) }
cat("\nPérdidas fuera de muestra (proxy: rendimiento al cuadrado), promedio diario\n")
cat("serie    QLIKE GARCH-t   QLIKE EWMA   QLIKE constante    ECM GARCH-t   ECM EWMA   ECM constante   DM QLIKE (GARCH - EWMA)\n")
for (j in seq_along(NOMBRES)) { p <- H[[j]]$p; s2 <- H[[j]]$s2; r <- R[, j]; o <- (ne + 1):n; e2 <- (r - p[1])^2
  he <- ewma_var(r); hc <- rep(var(R[1:ne, j]), n)
  q <- function(h) log(h[o]) + e2[o] / h[o]; m_ <- function(h) (e2[o] - h[o])^2
  d <- q(s2) - q(he); dc <- d - mean(d); S <- sum(dc^2); L <- 5; T <- length(d)
  for (k in 1:L) S <- S + 2 * (1 - k / (L + 1)) * sum(dc[(k + 1):T] * dc[1:(T - k)])
  cat(sprintf("%-6s  %12.4f  %11.4f  %14.4f   %12.2f  %9.2f  %13.2f   %10.2f\n", NOMBRES[j], mean(q(s2)), mean(q(he)), mean(q(hc)), mean(m_(s2)), mean(m_(he)), mean(m_(hc)), mean(d) / sqrt(S / T^2))) }
