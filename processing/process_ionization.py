import pandas as pd

pauling = pd.read_csv("../data/ionization_energy.csv")
dft = pd.read_csv("../data/dft.csv")

a_vals = dft["A"]
b_vals = dft["B"]

d = {}

for index, row in pauling.iterrows():
    d[row["Element"]] = row["Energy"]

en = pd.DataFrame({"A Ionization Energy":[0 for i in range(len(a_vals))],"B Ionization Energy":[0 for i in range(len(a_vals))]})

count = 0
for index, row in dft.iterrows():
    a = a_vals.iloc[index]
    b = b_vals.iloc[index]
    en["A Ionization Energy"].iloc[index] = d[a]
    en["B Ionization Energy"].iloc[index] = d[b]

print(en)

dft["A Ionization Energy"] = en["A Ionization Energy"]
dft["B Ionization Energy"] = en["B Ionization Energy"]
dft.to_csv("../data/dft_ionization_energy.csv")
    