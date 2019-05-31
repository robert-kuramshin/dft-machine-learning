import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import GradientBoostingRegressor

#read data
test = pd.read_csv("../data/battery/test.csv")
train = pd.read_csv("../data/battery/train.csv")

#specify feature column names
feature_cols = [
'HOMO (eV)',
'LUMO (eV)',
'EA (eV)',
'# C',
'# B',
'#O',
'Band Gap',
'#Li',
'# H',
'No. of Aromatic Rings',
]


feature_names = [
'HOMO (eV)',
'LUMO (eV)',
'EA (eV)',
'# C',
'# B',
'#O',
'Band Gap',
'#Li',
'# H',
'No. of Aromatic Rings',
]

#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train['RP (V) - DFT']

X_test = test.loc[:, feature_cols]
y_test = test['RP (V) - DFT']

scaler = StandardScaler()  

scaler.fit(X_train)  

X_train = scaler.transform(X_train)  

X_test = scaler.transform(X_test)  

params = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 1500, 'min_samples_leaf': 4, 'loss': 'ls', 'min_samples_split': 0.75}

clf = GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)

y_true, y_pred = y_test, clf.predict(X_test)
print("MSE with all features")

all_feature_mse = mean_squared_error(y_true, y_pred)
print(all_feature_mse)

print("Starting feature relevance analysis")

mse_arr = []
feature_arr = []

for feature in feature_cols:
    X_train = train.loc[:, feature_cols]
    y_train = train['RP (V) - DFT']

    X_test = test.loc[:, feature_cols]
    y_test = test['RP (V) - DFT']



    X_train_cut = X_train.drop(columns=feature)
    X_test_cut = X_test.drop(columns=feature)


    scaler = StandardScaler()  

    scaler.fit(X_train)  

    X_train = scaler.transform(X_train)  

    X_test = scaler.transform(X_test)  

    print(feature)
    params = {'alpha': 0.01, 'kernel': 'rbf'}
    clf = KernelRidge(**params)
    clf.fit(X_train_cut, y_train)

    y_true, y_pred = y_test, clf.predict(X_test_cut)
    print("MSE: ")
    mse = mean_squared_error(y_true, y_pred)
    print(mse)
    mse_arr.append(mse)
    feature_arr.append(feature)
print (mse_arr)

# ############################################################################
# Plot feature importance
feature_importance = mse_arr
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / max(feature_importance))
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
feature_arr = [feature_arr[i] for i in sorted_idx]
plt.yticks(pos, feature_arr)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
