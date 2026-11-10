# Capítulo 1, Ejercicio 1.5: media aritmética y geométrica, anualización y días de negociación por año
lee <- function(f) { x <- read.csv(file.path("datos/peru", f)); x$fecha <- as.Date(x$fecha); x }
P <- list(BVN = lee("bvn_yahoo.csv")[, c("fecha", "cierre_ajustado")],
          EPU = lee("epu_yahoo.csv")[, c("fecha", "cierre_ajustado")],
          SP500 = lee("gspc_yahoo.csv")[, c("fecha", "cierre")])
bvl <- subset(lee("bvl_indice_general_bcrp.csv"), fecha != as.Date("2024-06-21"))

medias <- vol <- NULL
for (k in names(P)) {
  x <- subset(P[[k]], fecha <= as.Date("2025-12-31")); p <- x[[2]]; a <- format(x$fecha, "%Y")
  pa <- tapply(p, a, function(v) tail(v, 1))                 # último precio de cada año
  Ra <- pa[-1] / pa[-length(pa)] - 1; T <- length(Ra)
  A <- mean(Ra); s2 <- mean((Ra - A)^2); G <- prod(1 + Ra)^(1 / T) - 1
  medias <- rbind(medias, c(T = T, A_pct = 100 * A, G_pct = 100 * G, A_s2_pct = 100 * (A - s2 / 2),
                            A_G_pb = 1e4 * (A - G), s2_2_pb = 1e4 * s2 / 2))
  rd <- diff(log(p)); dpa <- mean(table(a[-1]))
  vol <- rbind(vol, c(sd_diaria = 100 * sd(rd), raiz252 = 100 * sd(rd) * sqrt(252),
                      raiz_dias = 100 * sd(rd) * sqrt(dpa), sd_anual_obs = 100 * sd(log1p(Ra))))
}
rownames(medias) <- rownames(vol) <- names(P)
cat("(a) Medias de los rendimientos simples anuales, 2015-2025:\n"); print(round(medias, 2))
cat("\n(b) Volatilidad anualizada en % (rendimientos logarítmicos):\n"); print(round(vol, 2))
cuenta <- function(f) table(format(f, "%Y"))[as.character(2015:2025)]
cat("\n(c) Días con dato por año:\n")
d <- rbind(NYSE = cuenta(P$SP500$fecha), BVL_BCRP = cuenta(bvl$fecha)); print(d)
cat("Promedio:", round(rowMeans(d), 1), "\n")
