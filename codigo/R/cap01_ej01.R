# Capítulo 1, Ejercicio 1.1: dividendos de Credicorp (BAP) a partir del precio ajustado de Yahoo Finance
b <- read.csv("datos/peru/bap_yahoo.csv"); b$fecha <- as.Date(b$fecha)
P <- b$cierre; Pa <- b$cierre_ajustado; n <- length(P)
c_t   <- Pa / P                                  # c_t = P^a_t / P_t
salto <- c(NA, c_t[-1] / c_t[-n])                # c_t / c_{t-1} = 1 / f_t
ev    <- which(abs(salto - 1) > 1e-4)            # fechas ex-dividendo
P_ant <- P[ev - 1]; f <- 1 / salto[ev]
D     <- (1 - f) * P_ant                         # D_t = (1 - c_{t-1}/c_t) P_{t-1}
d     <- D / P_ant
Rtot  <- (P[ev] + D) / P_ant - 1
Ra    <- Pa[ev] / Pa[ev - 1] - 1
cat("Observaciones:", n, " (", format(b$fecha[1]), "a", format(b$fecha[n]), ")\n")
cat("Eventos detectados:", length(ev), "\n\n")
print(data.frame(fecha = b$fecha[ev], P_ant = round(P_ant, 2), f = round(f, 6),
                 D = round(D, 3), d_pct = round(100 * d, 2)), row.names = FALSE)
cat(sprintf("\nMáximo |error| en la Proposición 1.3: %.2e\n",
            max(abs((Ra - Rtot) - d / (1 - d) * Rtot))))
R <- c(NA, P[-1] / P[-n] - 1); RA <- c(NA, Pa[-1] / Pa[-n] - 1)
cat(sprintf("Máx |R^a - R| fuera de eventos: %.2e\n", max(abs(RA - R)[-c(1, ev)])))

acum <- function(x) x[length(x)] / x[1] - 1
dup  <- which(b$fecha == as.Date("2020-04-08"))  # registro duplicado de Yahoo
Pa_corr <- Pa * ifelse(seq_len(n) < dup, salto[dup], 1)
cat(sprintf("\nRendimiento acumulado con P (sin dividendos):       %8.2f %%\n", 100 * acum(P)))
cat(sprintf("Rendimiento acumulado con P^a de Yahoo:             %8.2f %%\n", 100 * acum(Pa)))
cat(sprintf("Rendimiento acumulado con P^a corregido (2020 x1):  %8.2f %%\n", 100 * acum(Pa_corr)))
anios <- as.numeric(b$fecha[n] - b$fecha[1]) / 365.25
cat(sprintf("Tasa anual compuesta con P:             %6.2f %%\n", 100 * ((1 + acum(P))^(1 / anios) - 1)))
cat(sprintf("Tasa anual compuesta con P^a corregido: %6.2f %%\n", 100 * ((1 + acum(Pa_corr))^(1 / anios) - 1)))
