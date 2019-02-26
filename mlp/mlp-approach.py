#doing some testing on the dft calculation database without collecting extra features. 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor

test_split_amount = 0.2 #20% of data is reserved for test

#read data
csv_data = pd.read_csv("../data/processed.csv")

#random shuffle
csv_data.sample(frac=1)

#specify feature column names
feature_cols = [
"Radius A [ang]",
"Radius B [ang]",
"Formation energy [eV/atom]",
"Volume per atom [A^3/atom]",
"Goldschmidt Tolerance Factor",
"A Electronegativity",
"B Electronegativity",
"C Electronegativity",
"A Ionization Energy",
"B Ionization Energy",
"C Ionization Energy",
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

#test train split
length = csv_data.shape[0]
train_size = int(length*(1-test_split_amount))

train = csv_data[:train_size]
test = csv_data[train_size:]

#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train["Band gap [eV]"]

X_test = test.loc[:, feature_cols]
y_test = np.array(test["Band gap [eV]"].values).astype(float)

scaler = StandardScaler()  

scaler.fit(X_train)  

X_train = scaler.transform(X_train)  

X_test = scaler.transform(X_test)  

#creating regressor and fitting data
params = {'learning_rate_init': 0.1, 'activation': 'logistic', 'hidden_layer_sizes': {100,}, 'max_iter': 4000}
reg = MLPRegressor(**params)

reg.fit(X_train, y_train)
mse = mean_squared_error(y_test, reg.predict(X_test))
print("MSE: %.4f" % mse)