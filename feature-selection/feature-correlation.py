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
'Radius A [ang]',
'Radius B [ang]',
'Formation energy [eV/atom]',
'Stability [eV/atom]',
'Magnetic moment [mu_B]',
'Volume per atom [A^3/atom]',
'a [ang]',
'b [ang]',
'c [ang]',
'gamma [deg]',
'Vacancy energy [eV/O atom]',
'Octahedral Factor',
'Tolerance Factor',
'A Ionization Energy',
'B Ionization Energy',
'A Electronegativity',
'B Electronegativity',
"A rp",
"A rd",
"B rp",
"B rd",
"B EA",
"B p total",
"B d total",
"B f total",
"A g",
"A p",
"B g",
]


feature_names = [
'Radius A [ang]',
'Radius B [ang]',
'Formation energy [eV/atom]',
'Stability [eV/atom]',
'Magnetic moment [mu_B]',
'Volume per atom [A^3/atom]',
'a [ang]',
'b [ang]',
'c [ang]',
'gamma [deg]',
'Vacancy energy [eV/O atom]',
'Octahedral Factor',
'Tolerance Factor',
'A Ionization Energy',
'B Ionization Energy',
'A Electronegativity',
'B Electronegativity',
"A p orbital radius",
"A d orbital radius",
"B p orbital radius",
"B d orbital radius",
"B Electron Affinity",
"B p orbital atoms",
"B d orbital atoms",
"B f orbital atoms",
"A group",
"A peroid",
"B group",
]


csv_data = csv_data.loc[:, feature_cols]

def plot_corr(df,size=10):

    corr = df.corr()
    corr.to_csv("corr.csv")
    fig, ax = plt.subplots(figsize=(size, size))
    clr = ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), feature_names, rotation='vertical')
    plt.yticks(range(len(corr.columns)), feature_names)
    plt.colorbar(clr)
    plt.show()
plot_corr(csv_data,16)
