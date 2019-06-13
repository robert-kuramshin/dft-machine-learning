import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

test = pd.read_csv("../data/battery/test.csv")
train = pd.read_csv("../data/battery/train.csv")

#specify feature column names
feature_cols = [
"HOMO (eV)",
"LUMO (eV)",
"EA (eV)",
"# C",
"# B",
"# O",
"HOMO-LUMO gap",
"# Li",
"# H",
"No. of Aromatic Rings",
"RP (V) - DFT",
]

feature_names = [
"a",
"b",
"c",
"d",
"e",
"f",
"g",
"h",
"i",
"j",
"k",
]

#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train['RP (V) - DFT']

X_test = test.loc[:, feature_cols]
y_test = test['RP (V) - DFT']

df = pd.DataFrame()
df["RP (V) - DFT"] = y_test
y_test = df

df = pd.DataFrame()
df["RP (V) - DFT"] = y_train
y_train = df

#normalizing 
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

#numpy array tp pandas dataframe
X_train = pd.DataFrame(X_train, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)

#transforming
transforms = [lambda x: np.power(x,2),lambda x: np.power(x,1/2),lambda x: np.log(1+x),lambda x: np.exp(x)]
transforms_format = ["({})^(2)","({})^(1/2)","log(1+({}))","n^({})"]
n_cols = len(X_train.columns)
n_trans = len(transforms)
for col in range(n_cols):
    for trans in range(n_trans):
        name = transforms_format[trans].format(X_train.columns[col])
        X_train[name] = 0 # add column
        X_train[name] = X_train.iloc[:,col]
        X_train[name] = X_train[name].transform(transforms[trans])

        name = transforms_format[trans].format(X_test.columns[col])
        X_test[name] = 0 # add column
        X_test[name] = X_test.iloc[:,col]
        X_test[name] = X_test[name].transform(transforms[trans])

X_test.to_csv("res/X_test_trans.csv")
X_train.to_csv("res/X_train_trans.csv")
y_test.to_csv("res/y_test.csv")
y_train.to_csv("res/y_train.csv")