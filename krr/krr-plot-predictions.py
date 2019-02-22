#doing some testing on the dft calculation database without collecting extra features. 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

test_split_amount = 0.1 #10% of data is reserved for test

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

#creating regressor and fitting data
params = {'alpha': 0.01, 'kernel': 'rbf'}
reg = KernelRidge(**params)

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