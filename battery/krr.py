#doing some testing on the dft calculation database without collecting extra features. 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#read data
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

#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train['RP (V) - DFT']

X_test = test.loc[:, feature_cols]
y_test = test['RP (V) - DFT']

scaler = StandardScaler()  

scaler.fit(X_train)  

X_train = scaler.transform(X_train)  

X_test = scaler.transform(X_test)  

#creating regressor and fitting data
params = {'kernel': 'rbf', 'alpha': 0.01}
reg = KernelRidge(**params)

reg.fit(X_train, y_train)

predicted = reg.predict(X_test)

y = y_test

print(r2_score(y, predicted))
mse = mean_squared_error(y_test, reg.predict(X_test))
print("MSE: %.4f" % mse)

print(y)
print(predicted)

fig,ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

test = pd.DataFrame(y_train)
test["predicted"] = reg.predict(X_train)
test.to_csv("test2.csv")