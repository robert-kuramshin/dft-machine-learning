import pandas as pd

radii = pd.read_csv("../data/orbital_radii.csv")
dft = pd.read_csv("../data/dft_pauling_electronegativity.csv")

a_vals = dft["A"]
b_vals = dft["B"]

d = {}

for index, row in radii.iterrows():
    d[row["Atom"]] = row

en = pd.DataFrame({"A rs":[0 for i in range(len(a_vals))],
                    "A rp":[0 for i in range(len(a_vals))],
                    "A rd":[0 for i in range(len(a_vals))],
                    "B rs":[0 for i in range(len(b_vals))],
                    "B rp":[0 for i in range(len(b_vals))],
                    "B rd":[0 for i in range(len(b_vals))]
                    })

count = 0
for index, row in dft.iterrows():
    a = a_vals.iloc[index]
    b = b_vals.iloc[index]
    if (a not in d.keys()):
        print (a)
    elif (b not in d.keys()):
        print (b)
    else: 
        en["A rs"].iloc[index] = d[a]["rs"]
        en["A rp"].iloc[index] = d[a]["rp"]
        en["A rd"].iloc[index] = d[a]["rd"]
        en["B rs"].iloc[index] = d[b]["rs"]
        en["B rp"].iloc[index] = d[b]["rp"]
        en["B rd"].iloc[index] = d[b]["rd"]

print(en)

dft["A rs"] = en["A rs"]
dft["A rp"] = en["A rp"]
dft["A rd"] = en["A rd"]
dft["B rs"] = en["B rs"]
dft["B rp"] = en["B rp"]
dft["B rd"] = en["B rd"]
dft.to_csv("../data/dft_atomic_radii.csv")
    