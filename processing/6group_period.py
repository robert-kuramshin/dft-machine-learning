import pandas as pd

radii = pd.read_csv("../data/group_period.csv")
dft = pd.read_csv("../data/dft_orbital_electrons.csv")

a_vals = dft["A"]
b_vals = dft["B"]

d = {}

for index, row in radii.iterrows():
    d[row["atom"]] = row

en = pd.DataFrame({"A g":[0 for i in range(len(a_vals))],
                    "A p":[0 for i in range(len(a_vals))],
                    "B g":[0 for i in range(len(b_vals))],
                    "B p":[0 for i in range(len(b_vals))]
                    })

count = 0
for index, row in dft.iterrows():
    a = a_vals.iloc[index]
    b = b_vals.iloc[index]

    en["A g"].iloc[index] = d[a]["group"]
    en["A p"].iloc[index] = d[a]["period"]
    en["B g"].iloc[index] = d[b]["group"]
    en["B p"].iloc[index] = d[b]["period"]

print(en)

dft["A g"] = en["A g"]
dft["A p"] = en["A p"]
dft["B g"] = en["B g"]
dft["B p"] = en["B p"]
dft.to_csv("../data/dft_group_period.csv")
    