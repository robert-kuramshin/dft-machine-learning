#doing some testing on the dft calculation database without collecting extra features. 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

test_split_amount = 0.1 #20% of data is reserved for test

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
params = {'n_estimators': 500, 'max_depth': 12, 'min_samples_split': 3,
          'learning_rate': 0.2, 'loss': 'ls'}
reg = GradientBoostingRegressor(**params)

reg.fit(X_train, y_train)
mse = mean_squared_error(y_test, reg.predict(X_test))
print("MSE: %.4f" % mse)

# #############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# #############################################################################
# Plot feature importance
feature_importance = reg.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
feature_cols = [feature_cols[i] for i in sorted_idx]
plt.yticks(pos, feature_cols)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
