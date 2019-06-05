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
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from scipy import stats

#read data
test = pd.read_csv("../data/battery/test.csv")
train = pd.read_csv("../data/battery/train.csv")

#specify feature column names
feature_cols = [
'HOMO (eV)',
'LUMO (eV)',
'EA (eV)',
'#O',
'# H',
'Band Gap',
'#Li',
'No. of Aromatic Rings',
]


feature_names = [
'HOMO (eV)',
'LUMO (eV)',
'EA (eV)',
'#O',
'# H',
'Band Gap',
'#Li',
'No. of Aromatic Rings',
]

#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train['RP (V) - DFT']

X_test = test.loc[:, feature_cols]
y_test = test['RP (V) - DFT']

print(X_train)
print(y_train)

scaler = StandardScaler()  

scaler.fit(X_train)  

X_train = scaler.transform(X_train)  

X_test = scaler.transform(X_test)  

#creating regressor and fitting data
params = {'alpha': 0.01, 'kernel': 'rbf'}
reg = KernelRidge(**params)

reg.fit(X_train, y_train)
mse = mean_squared_error(y_test, reg.predict(X_test))
print("MSE: %.4f" % mse)

print(r2_score(y_test,  reg.predict(X_test)))

print(reg.predict(X_test))

y = reg.predict(X_test)
x = y_test

print (y)
print(x)


fig = plt.figure(figsize=(2.2,2.2), dpi=300)
ax = plt.subplot(111)

ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(1))


#regression part
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

line = slope*x+intercept
plt.plot(x, line, 'r', label='y  = {:.2f}x+{:.2f}\nr2 = {:.4f}'.format(slope,intercept,r_value**2))
#end

plt.scatter(x,y, color="k", s=3.5)
plt.legend(fontsize=9)

plt.xlabel('RP (V) - DFT')
plt.ylabel('RP (V) - KRR Predicted')
plt.title('Test Dataset')

y = reg.predict(X_train)
x = y_train


fig = plt.figure(figsize=(2.2,2.2), dpi=300)
ax = plt.subplot(111)


ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(1))


#regression part
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

line = slope*x+intercept
plt.plot(x, line, 'r', label='y  = {:.2f}x+{:.2f}\nr2 = {:.4f}'.format(slope,intercept,r_value**2))
#end

plt.scatter(x,y, color="k", s=3.5)
plt.legend(fontsize=9)

plt.xlabel('RP (V) - DFT')
plt.ylabel('RP (V) - KRR Predicted')
plt.title('Train Dataset')

plt.show()