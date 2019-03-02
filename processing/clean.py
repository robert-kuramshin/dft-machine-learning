import pandas as pd
import numpy as np

csv_data = pd.read_csv("../data/processed.csv")

feature_cols = [
"Radius A [ang]",
"Radius B [ang]",
"Formation energy [eV/atom]",
"Volume per atom [A^3/atom]",
"Goldschmidt Tolerance Factor",
"A Electronegativity",
"B Electronegativity",
"A Ionization Energy",
"B Ionization Energy",
"Octahedral Factor",
"Tolerance Factor",
]

#remove rows with missing feature values
for feature in feature_cols:
    csv_data = csv_data[csv_data[feature] != "-"]

csv_data = csv_data[csv_data["Band gap [eV]"] != "0"]

test_split_amount = 0.2 #20% of data is reserved for test

csv_data = csv_data.sample(frac=1)

#test train split
length = csv_data.shape[0]
train_size = int(length*(1-test_split_amount))

train = csv_data.iloc[:train_size,:]
test = csv_data.iloc[train_size:,:]

print(train_size)
print(train.shape[0])
print(test.shape[0])

train.to_csv("../data/train.csv")
test.to_csv("../data/test.csv")