# Capítulo 12, Ejercicio 12.2: DCC en dos etapas para el S&P 500 y el Brent
source("codigo/R/cap12_funciones.R")
P <- panel_diario(); f <- P$fecha; e1 <- etapa1(P$R[, c(1, 3)])
cat("Etapa 1, GARCH(1,1) normal:\n")
for (i in 1:2) { p <- e1$modelos[[i]]$par
  cat(sprintf("  %-6s mu = %.4f  omega = %.4f  alpha = %.4f  beta = %.4f  log-ver = %.2f\n", c("S&P", "Brent")[i],
              p["mu"], p["omega"], p["alpha"], p["beta"], e1$modelos[[i]]$ll)) }
m <- dcc_fit(e1$Z)
cat(sprintf("\nEtapa 2, DCC(1,1): a = %.4f (ee %.4f), b = %.4f (ee %.4f); a + b = %.4f; vida media = %.0f días\n",
            m$a, m$ee[1], m$b, m$ee[2], m$a + m$b, log(0.5) / log(m$a + m$b)))
cat(sprintf("log-verosimilitud de correlación: DCC = %.2f; CCC = %.2f; LR = %.2f (p = %.2e, 2 g.l.)\n",
            m$ll, m$ll_ccc, 2 * (m$ll - m$ll_ccc), pchisq(2 * (m$ll - m$ll_ccc), 2, lower.tail = FALSE)))
cat(sprintf("correlación constante (CCC) de los residuos estandarizados = %.4f\n", m$Rc[1, 2]))
rho <- m$R[, 1, 2]
cat(sprintf("correlación DCC: mín. %.3f (%s), máx. %.3f (%s)\n", min(rho), f[which.min(rho)], max(rho), f[which.max(rho)]))
cat("promedio anual de la correlación DCC:\n")
for (y in c(2000, 2003, 2006, 2008, 2009, 2012, 2015, 2018)) cat(sprintf("  %d: %6.3f\n", y, mean(rho[substr(f, 1, 4) == y])))
