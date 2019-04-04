import pandas as pd
import numpy as np

ea = pd.read_csv("../data/electron_affinity.csv")
dft = pd.read_csv("../data/dft_atomic_radii.csv")

b_vals = dft["B"]

d = {}

for index, row in ea.iterrows():
    d[row["atom"]] = row

en = pd.DataFrame({"B EA":[0 for i in range(len(b_vals))],
                    })

count = 0
for index, row in dft.iterrows():
    b = b_vals.iloc[index]
    if (b not in d.keys()):
        print (b)
        en["B EA"].iloc[index] = np.nan
    else: 
        en["B EA"].iloc[index] = d[b]["ea"]

print(en)

dft["B EA"] = en["B EA"]
dft.to_csv("../data/dft_electron_affinity.csv")
    