#doing some testing on the dft calculation database without collecting extra features. 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor

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
"A Electronegativity",
"B Electronegativity",
"A Ionization Energy",
"B Ionization Energy",
"Octahedral Factor",
"Tolerance Factor",
]

#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train["Band gap [eV]"]

X_test = test.loc[:, feature_cols]
y_test = test["Band gap [eV]"]

scaler = StandardScaler()  

scaler.fit(X_train)  

X_train = scaler.transform(X_train)  

X_test = scaler.transform(X_test)  

#creating regressor and fitting data
params = {'max_iter': 5000, 'activation': 'relu', 'solver': 'lbfgs', 'learning_rate_init': 0.1, 'hidden_layer_sizes': {10}}
reg = MLPRegressor(**params)

reg.fit(X_train, y_train)

predicted = reg.predict(X_test)

y = y_test

print(r2_score(y, predicted))

fig,ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()