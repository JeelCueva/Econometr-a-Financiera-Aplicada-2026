# Capítulo 20: construcción del archivo de predictores mensuales (Shiller + French), 1927-2018
import pandas as pd, numpy as np
s = pd.read_csv("datos/internacional/sp500_mensual_shiller.csv"); s["mes"] = s["Date"].str[:7]
s = s[(s["Dividend"] > 0) & (s["Earnings"] > 0) & (s["Consumer Price Index"] > 0)]
fr = pd.read_csv("datos/internacional/french_mercado_mensual_1926_2018.csv"); fr.columns = ["mes", "MktRF", "SMB", "HML", "RF"]
d = s.merge(fr[["mes", "MktRF", "RF"]], on="mes", how="inner").reset_index(drop=True)
P, D, E, C = d["SP500"], d["Dividend"], d["Earnings"], d["Consumer Price Index"]
out = pd.DataFrame({"mes": d["mes"]})
out["dp"] = np.log(D / P); out["ep"] = np.log(E / P); out["de"] = np.log(D / E)
out["cape"] = np.log(d["PE10"].replace(0, np.nan)); out["lty"] = d["Long Interest Rate"]
out["tbl"] = 12 * d["RF"]; out["tms"] = out["lty"] - out["tbl"]
out["infl"] = (100 * (C / C.shift(12) - 1)).shift(1)
mk = d["MktRF"] + d["RF"]; out["mom12"] = mk.rolling(12).sum(); out["vol12"] = mk.rolling(12).std()
out["exceso"] = d["MktRF"]
out.dropna().reset_index(drop=True).round(6).to_csv("datos/internacional/predictores_mensuales_1927_2018.csv", index=False)
