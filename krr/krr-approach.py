#doing some testing on the dft calculation database without collecting extra features. 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt

#read data
test = pd.read_csv("../data/test.csv")
train = pd.read_csv("../data/train.csv")


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


feature_names = [
"Radius A",
"Radius B",
"Formation energy",
"Volume per atom",
"Goldschmidt Tolerance Factor",
"A EN",
"B EN",
"C EN",
"A IE",
"B IE",
"C IE",
]

#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train["Band gap [eV]"]

X_test = test.loc[:, feature_cols]
y_test = test["Band gap [eV]"]

print(X_train)
print(y_train)

scaler = StandardScaler()  

scaler.fit(X_train)  

X_train = scaler.transform(X_train)  

X_test = scaler.transform(X_test)  

#creating regressor and fitting data
params = {'alpha': 0.01, 'kernel': 'rbf'}
reg = KernelRidge(**params)

reg.fit(X_train, y_train)
mse = mean_squared_error(y_test, reg.predict(X_test))
print("MSE: %.4f" % mse)