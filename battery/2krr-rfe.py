#krr based recursive feature elimination results
import numpy as np
import pandas as pd

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

X_train = pd.read_csv("res/X_train_corr.csv",index_col=0)
X_test = pd.read_csv("res/X_test_corr.csv",index_col=0)
y_train = pd.read_csv("res/y_train.csv",index_col=0)
y_test = pd.read_csv("res/y_test.csv",index_col=0)

validation_ratio = 0.2
dataset_size = len(X_train)

X_train = train.loc[validation_ratio*dataset_size:, feature_cols]
y_train = train.loc[validation_ratio*dataset_size:, ['RP (V) - DFT']]

X_val = train.loc[:validation_ratio*dataset_size, feature_cols]
y_val = train.loc[:validation_ratio*dataset_size, ['RP (V) - DFT']]

#creating regressor and fitting data
tuned_parameters = [{'kernel':["linear","rbf"],'alpha': [0.001,0.01,0.1,0.5,1.0]}]
          
reg = GridSearchCV(KernelRidge(), tuned_parameters, cv=5,
                scoring='neg_mean_squared_error')

mse = []
r2 = []
feature = []

reg.fit(X_train, y_train.values.ravel())

predicted = reg.predict(X_val)

r2.append(r2_score(y_val, predicted))
mse.append(mean_squared_error(y_val, predicted))
feature.append("None")
print("----------------------------------")
print(mse[-1])
print(feature_cols[0])
feature.append(feature_cols[0])
feature_cols.remove(feature_cols[0])


while (feature_cols):
    X_train = X_train.loc[:, feature_cols]
    X_val = X_val.loc[:, feature_cols]

    reg = GridSearchCV(KernelRidge(), tuned_parameters, cv=5,
                    scoring='neg_mean_squared_error')

    reg.fit(X_train, y_train.values.ravel())
    predicted = reg.predict(X_val)

    r2.append(r2_score(y_val, predicted))
    mse.append(mean_squared_error(y_val, predicted))
    print("----------------------------------")
    print(mse[-1])
    print(feature_cols[0])
    if(len(feature_cols)>1):
        feature.append(feature_cols[0])
    feature_cols.remove(feature_cols[0])

