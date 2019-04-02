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
test = pd.read_csv("../data/test.csv")
train = pd.read_csv("../data/train.csv")


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
'alpha [deg]',
'beta [deg]',
'gamma [deg]',
'Vacancy energy [eV/O atom]',
'Octahedral Factor',
'Tolerance Factor',
'A Ionization Energy',
'B Ionization Energy',
'A Electronegativity',
'B Electronegativity',
'Goldschmidt Tolerance Factor'
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

#creating regressor and fitting data
params = {'loss': 'ls', 'max_depth': 4, 'learning_rate': 0.1, 'min_samples_split': 3, 'n_estimators': 500}

reg = GradientBoostingRegressor(**params)

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