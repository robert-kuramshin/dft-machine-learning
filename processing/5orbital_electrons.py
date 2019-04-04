import pandas as pd
import numpy as np

ea = pd.read_csv("../data/orbital-electron.csv")
dft = pd.read_csv("../data/dft_electron_affinity.csv")

b_vals = dft["B"]

d = {}

for index, row in ea.iterrows():
    d[row["atom"]] = row

en = pd.DataFrame({"B s total":[0 for i in range(len(b_vals))],"B p total":[0 for i in range(len(b_vals))],"B d total":[0 for i in range(len(b_vals))],"B f total":[0 for i in range(len(b_vals))],
                    })

count = 0
for index, row in dft.iterrows():
    b = b_vals.iloc[index]
    en["B s total"].iloc[index] = d[b]["s total"]
    en["B p total"].iloc[index] = d[b]["p total"]
    en["B d total"].iloc[index] = d[b]["d total"]
    en["B f total"].iloc[index] = d[b]["f total"]

print(en)

dft["B s total"] = en["B s total"]
dft["B p total"] = en["B p total"]
dft["B d total"] = en["B d total"]
dft["B f total"] = en["B f total"]

dft.to_csv("../data/dft_orbital_electrons.csv")
    