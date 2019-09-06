#initial model test
#initial feature importance 

import numpy as np
import pandas as pd

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pickle

#read data
X_train = pd.read_csv("res/X_train_rfe.csv",index_col=0)
X_test = pd.read_csv("res/X_test_rfe.csv",index_col=0)
y_train = pd.read_csv("res/y_train.csv",index_col=0)
y_test = pd.read_csv("res/y_test.csv",index_col=0)

#tuned_parameters = [{'kernel':["linear","rbf"],'alpha': np.logspace(-3,0,100)}]
tuned_parameters = [{'kernel':["linear","rbf"],'alpha': [0.01]}]

clf = GridSearchCV(KernelRidge(), tuned_parameters, cv=5)
clf.fit(X_train, y_train)

print(clf.best_params_)

y_pred = clf.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

np.savetxt("res/krr_compound_res.csv", y_pred, delimiter=",")

res_test = pd.DataFrame(index=y_train.index )
res_test["y"] = y_train
res_test["pred(y)"] = clf.predict(X_train)
res_test.to_csv("res/krr_compound_res_train.csv")

pickle.dump(clf, open("models/3krr.sav", 'wb'))