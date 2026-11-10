# Capítulo 5: funciones para pruebas de raíz unitaria (programadas desde las fórmulas del texto)

# Regresión de Dickey-Fuller aumentada:
#   dy_t = [constante] [+ tendencia] + gamma * y_{t-1} + sum_{j=1}^{p} delta_j dy_{t-j} + e_t
# 'inicio' permite fijar la muestra común al comparar rezagos.
adf_reg <- function(y, p, det = "c", inicio = p + 1) {
  dy <- diff(y); k <- inicio:length(dy)                 # filas de la regresión
  X <- cbind(ylag = y[k])
  if (p > 0) X <- cbind(X, sapply(1:p, function(j) dy[k - j]))
  if (det %in% c("c", "ct")) X <- cbind(X, const = 1)
  if (det == "ct") X <- cbind(X, tend = k)
  m <- lm.fit(X, dy[k]); e <- m$residuals; n <- length(e)
  V <- sum(e^2) / (n - ncol(X)) * chol2inv(qr.R(m$qr))
  list(gamma = m$coefficients[1], t = m$coefficients[1] / sqrt(V[1, 1]),
       ssr = sum(e^2), n = n, kcoef = ncol(X), coef = m$coefficients, ee = sqrt(diag(V)))
}
# Prueba ADF con número de rezagos elegido por AIC en una muestra común
adf <- function(y, det = "c", pmax = floor(12 * (length(y) / 100)^(1 / 4))) {
  aic <- sapply(0:pmax, function(p) {
    r <- adf_reg(y, p, det, inicio = pmax + 1)
    r$n * log(r$ssr / r$n) + 2 * r$kcoef
  })
  p <- which.min(aic) - 1
  c(adf_reg(y, p, det), p = p)
}
# Prueba KPSS con núcleo de Bartlett y l rezagos
kpss <- function(y, det = "c", l = floor(12 * (length(y) / 100)^(1 / 4))) {
  T <- length(y); tt <- 1:T
  e <- if (det == "c") y - mean(y) else resid(lm(y ~ tt))
  S <- cumsum(e)
  lrv <- sum(e^2) / T
  for (j in 1:l) lrv <- lrv + 2 * (1 - j / (l + 1)) * sum(e[(j + 1):T] * e[1:(T - j)]) / T
  c(eta = sum(S^2) / (T^2 * lrv), l = l)
}
# Valores críticos asintóticos (MacKinnon, 2010; Kwiatkowski et al., 1992)
vc_adf  <- list(c = c("1%" = -3.43, "5%" = -2.86, "10%" = -2.57),
                ct = c("1%" = -3.96, "5%" = -3.41, "10%" = -3.13))
vc_kpss <- list(c = c("10%" = 0.347, "5%" = 0.463, "1%" = 0.739),
                ct = c("10%" = 0.119, "5%" = 0.146, "1%" = 0.216))
