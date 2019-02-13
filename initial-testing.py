#doing some testing on the dft calculation database without collecting extra features. 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

test_split_amount = 0.2 #20% of data is reserved for test

#read data
csv_data = pd.read_csv("data/dft.csv")

#random shuffle
csv_data.sample(frac=1)

#specify feature column names
feature_cols = [
"Radius A [ang]",
"Radius B [ang]",
"Formation energy [eV/atom]",
"Stability [eV/atom]",
"Magnetic moment [mu_B]",
"Volume per atom [A^3/atom]",
"a [ang]",
"b [ang]",
"c [ang]",
"alpha [deg]",
"beta [deg]",
"gamma [deg]",
"Vacancy energy [eV/O atom]"
]

#remove rows with missing feature values
for feature in feature_cols:
    csv_data = csv_data[csv_data[feature] != "-"]

#removing elements with band gap of 0
csv_data = csv_data[csv_data["Band gap [eV]"] != "0.000"]

#test train split
length = csv_data.shape[0]
train_size = int(length*(1-test_split_amount))

train = csv_data[:train_size]
test = csv_data[train_size:]

#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train["Band gap [eV]"]

X_test = test.loc[:, feature_cols]
y_test = test["Band gap [eV]"]

#creating regressor and fitting data
reg = KernelRidge()

reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

res = mean_squared_error(y_test, y_pred)

print("Mean Squared Error: " + str(res))

plt.scatter(y_pred, y_test)
plt.show()