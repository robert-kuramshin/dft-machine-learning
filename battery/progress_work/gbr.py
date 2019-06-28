#doing some testing on the dft calculation database without collecting extra features. 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#read data
test = pd.read_csv("../../data/battery/test.csv")
train = pd.read_csv("../../data/battery/train.csv")


#specify feature column names
test = pd.read_csv("../../data/battery/test.csv")
train = pd.read_csv("../../data/battery/train.csv")

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

#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train['RP (V) - DFT']

X_test = test.loc[:, feature_cols]
y_test = test['RP (V) - DFT']


#creating regressor and fitting data
params = {'loss': 'ls', 'min_samples_leaf': 4, 'n_estimators': 500, 'min_samples_split': 5, 'learning_rate': 0.1, 'max_depth': 3}

reg = GradientBoostingRegressor(**params)

reg.fit(X_train, y_train)
predicted = reg.predict(X_test)
y = y_test

print(r2_score(y, predicted))
mse = mean_squared_error(y_test, reg.predict(X_test))
print("MSE: %.4f" % mse)


print(y)
print(predicted)

test = pd.DataFrame(y_train)
test["predicted"] = reg.predict(X_train)
test.to_csv("test1.csv")