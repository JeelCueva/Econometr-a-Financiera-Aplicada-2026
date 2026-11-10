# Capítulo 8, Ejercicio 8.3: GARCH(1,1) con innovaciones t de Student y diagnóstico de residuos estandarizados
source("codigo/R/cap08_funciones.R")
b <- read.csv("datos/internacional/brent_diario.csv"); r <- 100 * diff(log(b$Price)); T <- length(r)
mn <- garch_fit(r, "normal"); mt <- garch_fit(r, "t")
cat("GARCH(1,1)-t:\n")
print(round(cbind(coef = mt$par, ee_rob = mt$ee_rob, t_rob = round(mt$par / mt$ee_rob, 2)), 4))
crit <- function(ll, k) c(ll = ll, AIC = -2 * ll + 2 * k, BIC = -2 * ll + k * log(T))
cat("\n             log-ver        AIC        BIC\n")
cat(sprintf("normal  %12.2f %10.2f %10.2f\n", crit(mn$ll, 4)[1], crit(mn$ll, 4)[2], crit(mn$ll, 4)[3]))
cat(sprintf("t       %12.2f %10.2f %10.2f\n", crit(mt$ll, 5)[1], crit(mt$ll, 5)[2], crit(mt$ll, 5)[3]))
pers <- mt$par["alpha"] + mt$par["beta"]
cat(sprintf("\npersistencia = %.5f; vida media = %.1f días; curtosis de la t estandarizada = %.2f\n",
            pers, log(0.5) / log(pers), 3 + 6 / (mt$par["nu"] - 4)))
diagn <- function(x, nm) {
  e <- x - mean(x); s <- mean(e^3) / mean(e^2)^1.5; k <- mean(e^4) / mean(e^2)^2
  jb <- length(x) / 6 * (s^2 + (k - 3)^2 / 4)
  q1 <- Box.test(x, 10, type = "Ljung-Box")$statistic; q2 <- Box.test(e^2, 10, type = "Ljung-Box")$statistic
  cat(sprintf("%-22s asim. %6.2f  curt. %7.2f  JB %10.0f  Q(10) z %6.2f (p %.3f)  Q(10) z^2 %8.2f (p %.3f)\n",
              nm, s, k, jb, q1, pchisq(q1, 10, lower.tail = FALSE), q2, pchisq(q2, 10, lower.tail = FALSE)))
}
cat("\nDiagnóstico:\n")
diagn(r, "rendimientos r_t")
diagn((r - mt$par["mu"]) / sqrt(mt$s2), "residuos z_t (modelo t)")
z <- (r - mt$par["mu"]) / sqrt(mt$s2); nu <- mt$par["nu"]
cat(sprintf("\nproporción |z| > 4: observada %.5f; normal %.5f; t(nu) estandarizada %.5f\n",
            mean(abs(z) > 4), 2 * pnorm(-4), 2 * pt(-4 * sqrt(nu / (nu - 2)), nu)))
