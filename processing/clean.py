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

#remove rows with missing feature values
for feature in feature_cols:
    csv_data = csv_data[csv_data[feature] != "-"]

csv_data = csv_data[csv_data["Band gap [eV]"] != "0"]

csv_data.to_csv("../data/processed.csv")