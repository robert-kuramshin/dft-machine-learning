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

#read data
X_train = pd.read_csv("res/X_train_corr.csv",index_col=0)
X_test = pd.read_csv("res/X_test_corr.csv",index_col=0)
y_train = pd.read_csv("res/y_train.csv",index_col=0)
y_test = pd.read_csv("res/y_test.csv",index_col=0)

tuned_parameters = [{'kernel':["linear","rbf"],'alpha': np.linspace(0,15,16)}]

clf = GridSearchCV(KernelRidge(), tuned_parameters, cv=5)
clf.fit(X_train, y_train)

print(clf.best_params_)

y_pred = clf.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

np.savetxt("res/krr_compound_res.csv", y_pred, delimiter=",")