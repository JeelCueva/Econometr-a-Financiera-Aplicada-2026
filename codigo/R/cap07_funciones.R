# Capítulo 7: funciones propias (razón de varianzas de Lo-MacKinlay y prueba de rachas)
vr_lm <- function(r, q) {
  nq <- length(r); mu <- mean(r)
  sa <- sum((r - mu)^2) / (nq - 1)                        # varianza de 1 periodo (insesgada)
  rq <- stats::filter(r, rep(1, q), sides = 1)[q:nq]      # sumas móviles de q periodos (traslapadas)
  m <- q * (nq - q + 1) * (1 - q / nq)                    # corrección de sesgo de Lo-MacKinlay
  vr <- sum((rq - q * mu)^2) / m / sa
  z <- (vr - 1) / sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * nq))
  e2 <- (r - mu)^2; den <- sum(e2)^2
  theta <- 0
  for (j in 1:(q - 1)) {
    delta <- nq * sum(e2[(j + 1):nq] * e2[1:(nq - j)]) / den
    theta <- theta + (2 * (q - j) / q)^2 * delta
  }
  c(VR = vr, z = z, zrob = (vr - 1) / sqrt(theta / nq))
}
rachas <- function(x) {                                    # x: vector lógico (TRUE = sube)
  n1 <- sum(x); n2 <- sum(!x); n <- n1 + n2
  R <- 1 + sum(x[-1] != x[-n])
  ER <- 2 * n1 * n2 / n + 1
  VR <- 2 * n1 * n2 * (2 * n1 * n2 - n) / (n^2 * (n - 1))
  c(n1 = n1, n2 = n2, R = R, ER = ER, z = (R - ER) / sqrt(VR))
}
