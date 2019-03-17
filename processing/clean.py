import pandas as pd
import numpy as np

csv_data = pd.read_csv("../data/dft_pauling_electronegativity.csv")

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

#do this first to make sure we don't set Nan bandgaps to zero
csv_data = csv_data[csv_data["Band gap [eV]"] != "0"]

#remove rows with missing feature values
csv_data = csv_data.replace('-', np.nan)

csv_data.dropna(subset=["Band gap [eV]"],inplace=True)

for feature in feature_cols:
    print(feature)
    csv_data[feature].fillna(csv_data[feature].astype('float32').mean(),inplace=True)

test_split_amount = 0.2 #20% of data is reserved for test

csv_data = csv_data.sample(frac=1)

#test train split
length = csv_data.shape[0]
train_size = int(length*(1-test_split_amount))

train = csv_data.iloc[:train_size,:]
test = csv_data.iloc[train_size:,:]

print(train.shape[0])
print(test.shape[0])

train = train.reset_index()
test = test.reset_index()

train.to_csv("../data/train.csv")
test.to_csv("../data/test.csv")