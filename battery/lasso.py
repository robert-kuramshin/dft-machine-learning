import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

X_train = pd.read_csv("res/X_train_comp.csv",index_col=0)
X_test = pd.read_csv("res/X_test_comp.csv",index_col=0)
y_train = pd.read_csv("res/y_train.csv",index_col=0)
y_test = pd.read_csv("res/y_test.csv",index_col=0)


tuned_parameters = [{'alpha': [0.0001,0.001,0.01,0.1,0.2,0.5],"max_iter":[10000000]}]
las = GridSearchCV(linear_model.Lasso(), tuned_parameters, cv=5, n_jobs=3)
las.fit(X_train, y_train)

mse = mean_squared_error(y_test, las.predict(X_test))
print("MSE: %.4f" % mse)

print(las.best_estimator_.coef_)
print(las.best_estimator_.intercept_)  

new_features = []
count = 0
for coef in las.best_estimator_.coef_:
    if(not coef == 0.0):
        new_features.append(X_train.columns[count])
    count+=1

print (new_features)

X_train = X_train[new_features]
X_test = X_test[new_features]

X_test.to_csv("res/X_test_lasso.csv")
X_train.to_csv("res/X_train_lasso.csv")