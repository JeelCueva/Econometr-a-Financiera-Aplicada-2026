# Capítulo 4, Ejercicio 4.1: heterocedasticidad en el modelo de mercado de Utils
d <- read.csv("datos/internacional/french_mensual_1949_2017.csv")
y <- d$Utils - d$RF; x <- d$MktRF; T <- length(y)
X <- cbind(1, x); k <- ncol(X)
XtXi <- solve(t(X) %*% X)
b <- XtXi %*% t(X) %*% y; e <- as.vector(y - X %*% b)

# Pruebas de Breusch-Pagan (e^2 sobre x) y de White (e^2 sobre x y x^2): LM = T * R2
lm_prueba <- function(Z) {
  u <- e^2; R2 <- summary(lm(u ~ Z))$r.squared
  LM <- T * R2; q <- NCOL(Z)
  c(LM = LM, gl = q, p = pchisq(LM, q, lower.tail = FALSE))
}
cat("Breusch-Pagan:", round(lm_prueba(x), 4), "\n")
cat("White:        ", round(lm_prueba(cbind(x, x^2)), 4), "\n\n")

# Errores estándar: clásico, HC0 y HC1 (matriz sándwich)
s2   <- sum(e^2) / (T - k)
V_cl <- s2 * XtXi
V_h0 <- XtXi %*% (t(X * e) %*% (X * e)) %*% XtXi
V_h1 <- V_h0 * T / (T - k)
ee <- sapply(list(clasico = V_cl, HC0 = V_h0, HC1 = V_h1), function(V) sqrt(diag(V)))
rownames(ee) <- c("alfa", "beta")
print(round(ee, 5))
cat("\nt(alfa = 0):", round(b[1] / ee[1, ], 3), "\n")
cat("t(beta = 1):", round((b[2] - 1) / ee[2, ], 3), "\n")
cat(sprintf("Cociente ee(beta) HC1/clásico = %.3f\n", ee[2, "HC1"] / ee[2, "clasico"]))
