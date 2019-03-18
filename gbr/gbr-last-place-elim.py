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
# feature_cols = [
# 'Radius A [ang]',
# 'Radius B [ang]',
# 'Formation energy [eV/atom]',
# 'Stability [eV/atom]',
# 'Magnetic moment [mu_B]',
# 'Volume per atom [A^3/atom]',
# 'a [ang]',
# 'b [ang]',
# 'c [ang]',
# 'alpha [deg]',
# 'beta [deg]',
# 'gamma [deg]',
# 'Vacancy energy [eV/O atom]',
# 'Octahedral Factor',
# 'Tolerance Factor',
# 'A Ionization Energy',
# 'B Ionization Energy',
# 'A Electronegativity',
# 'B Electronegativity',
# 'Goldschmidt Tolerance Factor'
# ]
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


#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train["Band gap [eV]"]

X_test = test.loc[:, feature_cols]
y_test = test["Band gap [eV]"]

#creating regressor and fitting data
params = {'loss': 'ls', 'max_depth': 4, 'learning_rate': 0.1, 'min_samples_split': 3, 'n_estimators': 500}
reg = GradientBoostingRegressor(**params)

mse = []
r2 = []
feature = []

X_train = train.loc[:, feature_cols]
X_test = test.loc[:, feature_cols]

reg.fit(X_train, y_train)
predicted = reg.predict(X_test)

r2.append(r2_score(y_test, predicted))
mse.append(mean_squared_error(y_test, predicted))
feature.append("None")
print(mse[-1])

feature_importance = reg.feature_importances_
print(feature_importance)
worst_feature = feature_cols[feature_importance.argmin()]
print("Eliminating: ",worst_feature)
feature_cols.remove(worst_feature)
feature.append(worst_feature)

while (feature_cols):
    X_train = train.loc[:, feature_cols]
    X_test = test.loc[:, feature_cols]

    reg.fit(X_train, y_train)
    predicted = reg.predict(X_test)

    r2.append(r2_score(y_test, predicted))
    mse.append(mean_squared_error(y_test, predicted))

    print(mse[-1])

    feature_importance = reg.feature_importances_
    worst_feature = feature_cols[feature_importance.argmin()]
    print("Eliminating: ",worst_feature)

    if(len(feature_cols)>1):
        feature.append(worst_feature)
    feature_cols.remove(worst_feature)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('MSE')
plt.plot(feature,mse, 'b-',
         label='Mean Square Error')
plt.xticks(feature, feature, rotation='vertical')
plt.legend(loc='upper right')
plt.xlabel('Eliminated Feature')
plt.ylabel('Mean Square Error')

plt.subplot(1, 2, 2)
plt.title('R2')
plt.plot(feature, r2, 'r-',
         label='R2 Value')
plt.legend(loc='upper right')
plt.xlabel('Eliminated Feature')
plt.xticks(feature, feature, rotation='vertical')
plt.ylabel('R2')

plt.show()