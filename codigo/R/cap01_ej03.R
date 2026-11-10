# Capítulo 1, Ejercicio 1.3: rendimiento en soles de Southern Copper (SCCO) con dos fuentes del tipo de cambio
lee <- function(f) { x <- read.csv(file.path("datos/peru", f)); x$fecha <- as.Date(x$fecha); x }
P  <- lee("scco_yahoo.csv")                    # cierre_ajustado en dólares
Sb <- lee("tc_interbancario_venta_bcrp.csv")   # BCRP PD04638PD
Sy <- lee("pen_x_yahoo_mensual.csv")           # Yahoo PEN=X, último dato de cada mes

cortes <- as.Date(c(paste0(2014:2025, "-12-31"), "2026-10-30"))
ultimo <- function(x, col, d) sapply(d, function(di) tail(x[[col]][x$fecha <= di], 1))
tab <- data.frame(P = ultimo(P, "cierre_ajustado", cortes),
                  S_bcrp = ultimo(Sb, "tc", cortes), S_yahoo = ultimo(Sy, "tc", cortes))
lr  <- apply(log(tab), 2, diff)
out <- data.frame(r_usd = lr[, "P"], dlnS_bcrp = lr[, "S_bcrp"], r_sol_bcrp = lr[, "P"] + lr[, "S_bcrp"],
                  dlnS_yahoo = lr[, "S_yahoo"], r_sol_yahoo = lr[, "P"] + lr[, "S_yahoo"],
                  row.names = c(2015:2025, "2026*"))
cat("Descomposición anual (%, logarítmica):  r_soles = r_dólares + dln S\n")
print(round(100 * out, 2))

V <- tab$P * tab$S_bcrp; k <- nrow(tab)
simple <- function(x) x[-1] / x[-k] - 1
Rs <- simple(V); Ru <- simple(tab$P); RS <- simple(tab$S_bcrp)
cat(sprintf("\nMáx |R_sol - (R_usd + R_S + R_usd*R_S)| = %.1e\n", max(abs(Rs - (Ru + RS + Ru * RS)))))
dS <- out$dlnS_bcrp - out$dlnS_yahoo
cat(sprintf("Diferencia entre fuentes en dln S (pp): media = %.3f, máx abs = %.3f\n",
            100 * mean(dS), 100 * max(abs(dS))))
tot <- colSums(out)
cat("\nPeríodo completo (dic-2014 a oct-2026), suma de logaritmos (%):\n")
cat(sprintf("   acción en dólares %7.2f | tipo de cambio BCRP %6.2f | en soles %7.2f\n",
            100 * tot["r_usd"], 100 * tot["dlnS_bcrp"], 100 * tot["r_sol_bcrp"]))
cat(sprintf("   rendimiento simple en soles: %.2f %%  (en dólares: %.2f %%)\n",
            100 * (exp(tot["r_sol_bcrp"]) - 1), 100 * (exp(tot["r_usd"]) - 1)))
