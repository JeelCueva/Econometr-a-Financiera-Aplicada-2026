# Capítulo 10, Ejercicio 10.3: respuestas al impulso del S&P 500 y del VIX; sensibilidad al orden de Cholesky
source("codigo/R/cap10_funciones.R")
s <- read.csv("datos/internacional/sp500_diario_yahoo.csv"); v <- read.csv("datos/internacional/vix_diario.csv")
d <- merge(s[, c("Date", "Adj.Close")], v[, c("DATE", "CLOSE")], by.x = "Date", by.y = "DATE")
Y <- cbind(r = 100 * diff(log(d$Adj.Close)), dv = 100 * diff(log(d$CLOSE)))
m <- var_mco(Y, 2); P <- t(chol(m$S))                     # chol() de R da la triangular superior
cat("Factor de Cholesky P (orden r, dv):\n"); print(round(P, 4))
th <- irf(m$A, 10, P)
cat("\nOrden (r, dv). Respuestas ortogonalizadas:\n  h    r<-r    dv<-r    r<-dv   dv<-dv\n")
for (h in c(0, 1, 2, 5, 10)) { T <- th[[h + 1]]
  cat(sprintf("%3d %7.4f %8.4f %8.4f %8.4f\n", h, T[1, 1], T[2, 1], T[1, 2], T[2, 2])) }
Yr <- Y[, 2:1]; mr <- var_mco(Yr, 2); thr <- irf(mr$A, 10, t(chol(mr$S)))
cat("\nOrden (dv, r). Respuestas (variables en el orden r, dv):\n  h    r<-r    dv<-r    r<-dv   dv<-dv\n")
for (h in c(0, 1, 2, 5, 10)) { T <- thr[[h + 1]][2:1, 2:1]
  cat(sprintf("%3d %7.4f %8.4f %8.4f %8.4f\n", h, T[1, 1], T[2, 1], T[1, 2], T[2, 2])) }
acum <- Reduce(`+`, th)
cat(sprintf("\nrespuesta acumulada (10 días) del VIX a un choque de +1 desv. est. en r: %.3f %%\n", acum[2, 1]))
b <- bootstrap_irf(Y, 2, 10, B = 500, semilla = 2026)
cat("Bandas bootstrap al 90 % (500 réplicas), orden (r, dv):\n")
for (h in c(0, 1, 2)) cat(sprintf("  h = %d: dv<-r [%.3f, %.3f]; r<-dv [%.4f, %.4f]\n", h,
                                  b$inf[h + 1, 2, 1], b$sup[h + 1, 2, 1], b$inf[h + 1, 1, 2], b$sup[h + 1, 1, 2]))
