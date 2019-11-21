import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler

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

X_train.to_csv("res/X_train_trans.csv")
y_test.to_csv("res/y_test.csv")
y_train.to_csv("res/y_train.csv")
#normalizing 
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

scaler = MaxAbsScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

#numpy array tp pandas dataframe
X_train = pd.DataFrame(X_train, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)

all_mean = X_train.mean()
all_min = X_train.min()
all_max = X_train.max()

f1 = "h"
f2 = "c"

s1 = (all_max[f1] - all_min[f1])/10
s2 = (all_max[f2] - all_min[f2])/10

v1 =all_min[f1]
v2 =all_min[f2]

n = pd.DataFrame()

l1 = []
l2 = []
for i in range(10):
    v2 = all_min[f2]
    for j in range(10):
        l1.append(v1)
        l2.append(v2)
        v2 += s2
    v1 += s1

n[f1] = l1
n[f2] = l2

for name in feature_names:
    if name is not f1 and name is not f2:
        n[name] = all_mean[name]

n.to_csv("res/step1.csv")

# transforms = [lambda x: np.power(x,2),lambda x: np.power(1+x,1/2),lambda x: np.log(2+x),lambda x: np.exp(x)]

X_test = pd.DataFrame()

cf1 = (1 + n["c"]).pow(1/2) + np.exp(n["c"]) + np.exp(n["b"])
cf2 = (1 + n["c"]).pow(1/2) + np.exp(n["c"]) + np.exp(n["i"])
cf3 = n["c"] + np.exp(n["c"]) + np.exp(n["i"])
cf4 = (1 + n["c"]).pow(1/2) + np.exp(n["c"]) + np.exp(n["g"])
cf5 = (1 + n["c"]).pow(1/2) + np.exp(n["c"]) + np.exp(n["h"])
cf6 = (1 + n["c"]).pow(1/2) + np.exp(n["c"]) + np.exp(n["e"])
cf7 = np.exp(n["c"]) + np.exp(n["i"]) + np.exp(n["j"])

X_test["cf1"] = cf1
X_test["cf2"] = cf2
X_test["cf3"] = cf3
X_test["cf4"] = cf4
X_test["cf5"] = cf5
X_test["cf6"] = cf6
X_test["cf7"] = cf7

# #since maxabs scaler was trained only on X_train, it's values may exceed [-1,1]
# #hence we need to fill NA values
# for feature in X_test.columns:
#     X_test[feature].fillna(X_test[feature].astype('float32').mean(),inplace=True)

X_test.to_csv("res/X_test_step.csv")
# X_train.to_csv("res/X_train_trans.csv")
# y_test.to_csv("res/y_test.csv")
# y_train.to_csv("res/y_train.csv")