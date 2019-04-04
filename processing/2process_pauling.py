import pandas as pd

pauling = pd.read_csv("../data/pauling_en.csv")
dft = pd.read_csv("../data/dft_ionization_energy.csv")

a_vals = dft["A"]
b_vals = dft["B"]

d = {}

for index, row in pauling.iterrows():
    d[row["Element"]] = row["EN"]

en = pd.DataFrame({"A Electronegativity":[0 for i in range(len(a_vals))],"B Electronegativity":[0 for i in range(len(a_vals))]})

count = 0
for index, row in dft.iterrows():
    a = a_vals.iloc[index]
    b = b_vals.iloc[index]
    en["A Electronegativity"].iloc[index] = d[a]
    en["B Electronegativity"].iloc[index] = d[b]

print(en)

dft["A Electronegativity"] = en["A Electronegativity"]
dft["B Electronegativity"] = en["B Electronegativity"]
dft.to_csv("../data/dft_pauling_electronegativity.csv")
    