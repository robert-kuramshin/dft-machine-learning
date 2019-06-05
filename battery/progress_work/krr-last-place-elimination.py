#doing some testing on the dft calculation database without collecting extra features. 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


#read data
test = pd.read_csv("../data/battery/test.csv")
train = pd.read_csv("../data/battery/train.csv")


#specify feature column names
feature_cols = [
'# C',
'#O',
'# H',
'No. of Aromatic Rings',
'Band Gap',
'# B',
'HOMO (eV)',
'LUMO (eV)',
'#Li',
'EA (eV)',
]


feature_names = [
'# C',
'#O',
'# H',
'No. of Aromatic Rings',
'Band Gap',
'# B',
'HOMO (eV)',
'LUMO (eV)',
'#Li',
'EA (eV)',
]

'LUMO (eV)',
'# C',
'# B',
'# H',
'No. of Aromatic Rings',
'Band Gap',
'EA (eV)',

#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train['RP (V) - DFT']

X_test = test.loc[:, feature_cols]
y_test = test['RP (V) - DFT']


#creating regressor and fitting data
tuned_parameters = [{'kernel':["linear","rbf"],'alpha': [0.001,0.01,0.1,0.5,1.0,1.5]}]
          
reg = GridSearchCV(KernelRidge(), tuned_parameters, cv=5,
                scoring='neg_mean_squared_error')

mse = []
r2 = []
feature = []

X_train = train.loc[:, feature_cols]
X_test = test.loc[:, feature_cols]

reg.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(reg.best_params_)

predicted = reg.predict(X_test)

r2.append(r2_score(y_test, predicted))
mse.append(mean_squared_error(y_test, predicted))
feature.append("None")

print(feature_cols[0])
feature.append(feature_cols[0])
feature_cols.remove(feature_cols[0])


while (feature_cols):

    X_train = train.loc[:, feature_cols]
    X_test = test.loc[:, feature_cols]

    reg = GridSearchCV(KernelRidge(), tuned_parameters, cv=5,
                    scoring='neg_mean_squared_error')

    reg.fit(X_train, y_train)
    predicted = reg.predict(X_test)

    print("Best parameters set found on development set:")
    print()
    print(reg.best_params_)

    r2.append(r2_score(y_test, predicted))
    mse.append(mean_squared_error(y_test, predicted))

    print(mse[-1])

    print(feature_cols[0])
    if(len(feature_cols)>1):
        feature.append(feature_cols[0])
    feature_cols.remove(feature_cols[0])

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