# Apéndice C: inventario de los archivos de datos/peru (número de filas, primera y última fecha y valor)
for (f in sort(list.files("datos/peru", pattern = "\\.csv$", full.names = TRUE))) {
  x <- read.csv(f, colClasses = "character"); n <- nrow(x)
  cat(sprintf("%-34s %5d  %s %10s  %s %10s\n", basename(f), n, x[1, 1], x[1, 2], x[n, 1], x[n, 2]))
}
