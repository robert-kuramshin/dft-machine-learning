#initial model test
#initial feature importance 

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#read data
test = pd.read_csv("../data/battery/test.csv")
train = pd.read_csv("../data/battery/train.csv")

#specify feature column names
feature_cols = [
"HOMO (eV)",
"LUMO (eV)",
"EA (eV)",
"# C",
"# B",
"# O",
"HOMO-LUMO gap",
"# Li",
"# H",
"No. of Aromatic Rings",
]

feature_names = [ 
"HOMO (eV)",
"LUMO (eV)",
"EA (eV)",
"# C",
"# B",
"# O",
"HOMO-LUMO gap",
"# Li",
"# H",
"No. of Aromatic Rings",
]

#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train['RP (V) - DFT']

X_test = test.loc[:, feature_cols]
y_test = test['RP (V) - DFT']


#normalizing 
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

tuned_parameters = [{'n_estimators': [100,200,500],
                     'max_depth': [1,2,3,4,5,6,7,8],
                     'min_samples_leaf':[1,2,3,4,5],
                    'min_samples_split': [0.5,1.0,2,3,4,5,6,7,8,9,10,11,12],
                    'learning_rate': [0.05,0.075,0.1,0.125,0.15,0.2],
               "loss":['ls','lad','huber']}]

clf = GridSearchCV(GradientBoostingRegressor(), tuned_parameters, cv=5,n_jobs=7)
clf.fit(X_train, y_train)
print(clf.best_params_)

y_pred = clf.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


np.savetxt("res/gbr_raw.csv", y_pred, delimiter=",")




