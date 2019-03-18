from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#read data
csv_data = pd.read_csv("../data/train.csv")

#specify feature column names
#specify feature column names
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

csv_data = csv_data.loc[:, feature_cols]

def plot_corr(df,size=10):

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    clr = ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns);
    plt.colorbar(clr)
    plt.show()
plot_corr(csv_data,16)
