import pandas as pd

pauling = pd.read_csv("data/ionization_energy.csv")
dft = pd.read_csv("data/dft_improved2.csv")

a_vals = dft["A"]
b_vals = dft["B"]
c_vals = dft["C"]

d = {}

for index, row in pauling.iterrows():
    d[row["Element"]] = row["Energy"]

en = pd.DataFrame({"A_IE":[0 for i in range(len(a_vals))],"B_IE":[0 for i in range(len(a_vals))],"C_IE":[0 for i in range(len(a_vals))]})

count = 0
for index, row in dft.iterrows():
    a = a_vals.iloc[index]
    b = b_vals.iloc[index]
    c = c_vals.iloc[index]
    en["A_IE"].iloc[index] = d[a]
    en["B_IE"].iloc[index] = d[b]
    en["C_IE"].iloc[index] = d[c]

print(en)

dft["A_IE"] = en["A_IE"]
dft["B_IE"] = en["B_IE"]
dft["C_IE"] = en["C_IE"]
dft.to_csv("data/dft_improved3.csv")
    