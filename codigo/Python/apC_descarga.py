# Apéndice C: descarga de las series del Capítulo 1 (API del BCRP y punto de acceso v8 de Yahoo Finance)
# Requiere conexión a internet; no se ejecuta al compilar el libro.
import numpy as np, pandas as pd, requests

MESES = {"Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
         "Jul": 7, "Ago": 8, "Set": 9, "Oct": 10, "Nov": 11, "Dic": 12}

def bcrp(codigo, inicio="2014-12-31", fin="2026-11-08"):
    a, b = pd.Timestamp(inicio), pd.Timestamp(fin)       # fechas diarias sin ceros: 2014-12-31
    url = (f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{codigo}/json/"
           f"{a.year}-{a.month}-{a.day}/{b.year}-{b.month}-{b.day}")
    j = requests.get(url, timeout=60).json()
    print(codigo, "->", j["config"]["series"][0]["name"])
    filas = []
    for p in j["periods"]:                                # p["name"] tiene la forma "31.Dic.14"
        d, m, y = p["name"].split(".")
        v = p["values"][0]
        filas.append((pd.Timestamp(2000 + int(y), MESES[m], int(d)), np.nan if v == "n.d." else float(v)))
    return pd.DataFrame(filas, columns=["fecha", "valor"]).dropna()

def yahoo(simbolo, p1=1419984000, p2=1794268800):     # 31-dic-2014 a 10-nov-2026 (segundos Unix)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{simbolo}"
    q = dict(period1=p1, period2=p2, interval="1d", events="div|split", includeAdjustedClose="true")
    r = requests.get(url, params=q, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).json()["chart"]["result"][0]
    fecha = pd.to_datetime(np.array(r["timestamp"]) + r["meta"]["gmtoffset"], unit="s").normalize()
    return pd.DataFrame({"fecha": fecha, "cierre": r["indicators"]["quote"][0]["close"],
                         "cierre_ajustado": r["indicators"]["adjclose"][0]["adjclose"]}).dropna()

if __name__ == "__main__":
    bcrp("PD38026MD").rename(columns={"valor": "indice"}).to_csv("datos/peru/bvl_indice_general_bcrp.csv", index=False)
    bcrp("PD04638PD").rename(columns={"valor": "tc"}).to_csv("datos/peru/tc_interbancario_venta_bcrp.csv", index=False)
    for s in ["BAP", "SCCO", "BVN", "EPU"]:
        yahoo(s).round(4).to_csv(f"datos/peru/{s.lower()}_yahoo.csv", index=False)
    yahoo("^GSPC")[["fecha", "cierre"]].to_csv("datos/peru/gspc_yahoo.csv", index=False)
