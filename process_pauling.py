import pandas as pd

pauling = pd.read_csv("data/pauling_en.csv")
dft = pd.read_csv("data/dft_improved.csv")

a_vals = dft["A"]
b_vals = dft["B"]
c_vals = dft["C"]

d = {}

for index, row in pauling.iterrows():
    d[row["Element"]] = row["EN"]

en = pd.DataFrame({"A_EN":[0 for i in range(len(a_vals))],"B_EN":[0 for i in range(len(a_vals))],"C_EN":[0 for i in range(len(a_vals))]})

count = 0
for index, row in dft.iterrows():
    a = a_vals.iloc[index]
    b = b_vals.iloc[index]
    c = c_vals.iloc[index]
    en["A_EN"].iloc[index] = d[a]
    en["B_EN"].iloc[index] = d[b]
    en["C_EN"].iloc[index] = d[c]

print(en)

dft["A_EN"] = en["A_EN"]
dft["B_EN"] = en["B_EN"]
dft["C_EN"] = en["C_EN"]
dft.to_csv("data/dft_improved2.csv")
    