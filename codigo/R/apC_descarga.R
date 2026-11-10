# Apéndice C: descarga de las series del Capítulo 1 (API del BCRP y punto de acceso v8 de Yahoo Finance)
# Requiere conexión a internet y el paquete jsonlite; no se ejecuta al compilar el libro.
library(jsonlite)
meses <- c(Ene = 1, Feb = 2, Mar = 3, Abr = 4, May = 5, Jun = 6, Jul = 7, Ago = 8,
           Set = 9, Oct = 10, Nov = 11, Dic = 12)
fd <- function(d) { d <- as.POSIXlt(as.Date(d)); sprintf("%d-%d-%d", d$year + 1900, d$mon + 1, d$mday) }

bcrp <- function(codigo, inicio = "2014-12-31", fin = "2026-11-08") {
  u <- sprintf("https://estadisticas.bcrp.gob.pe/estadisticas/series/api/%s/json/%s/%s",
               codigo, fd(inicio), fd(fin))
  j <- fromJSON(u)
  cat(codigo, "->", j$config$series$name, "\n")
  p <- strsplit(j$periods$name, ".", fixed = TRUE)            # "31.Dic.14"
  fecha <- as.Date(sapply(p, function(z) sprintf("20%s-%02d-%s", z[3], meses[z[2]], z[1])))
  v <- suppressWarnings(as.numeric(unlist(j$periods$values)))  # "n.d." -> NA
  na.omit(data.frame(fecha = fecha, valor = v))
}

yahoo <- function(simbolo, p1 = 1419984000, p2 = 1794268800) {
  u <- sprintf(paste0("https://query1.finance.yahoo.com/v8/finance/chart/%s?period1=%d&period2=%d",
                      "&interval=1d&events=div%%7Csplit&includeAdjustedClose=true"),
               URLencode(simbolo, reserved = TRUE), p1, p2)
  con <- url(u, headers = c("User-Agent" = "Mozilla/5.0"))
  r <- fromJSON(paste(readLines(con, warn = FALSE), collapse = ""))$chart$result
  fecha <- as.Date(as.POSIXct(r$timestamp[[1]] + r$meta$gmtoffset, origin = "1970-01-01", tz = "UTC"))
  na.omit(data.frame(fecha = fecha, cierre = r$indicators$quote[[1]]$close[[1]],
                     cierre_ajustado = r$indicators$adjclose[[1]]$adjclose[[1]]))
}

write.csv(setNames(bcrp("PD38026MD"), c("fecha", "indice")), "datos/peru/bvl_indice_general_bcrp.csv", row.names = FALSE)
write.csv(setNames(bcrp("PD04638PD"), c("fecha", "tc")), "datos/peru/tc_interbancario_venta_bcrp.csv", row.names = FALSE)
for (s in c("BAP", "SCCO", "BVN", "EPU"))
  write.csv(yahoo(s), sprintf("datos/peru/%s_yahoo.csv", tolower(s)), row.names = FALSE)
