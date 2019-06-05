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
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from scipy import stats

#read data
test = pd.read_csv("../data/battery/test.csv")
train = pd.read_csv("../data/battery/train.csv")

#specify feature column names
feature_cols = [
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

tuned_parameters = [{'kernel':["linear","rbf"],'alpha': [0.001,0.01,0.1,0.5,1.0,1.5]}]

clf = GridSearchCV(KernelRidge(), tuned_parameters, cv=5)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(mean_squared_error(y_true, y_pred))

print()


y = clf.predict(X_test)
x = y_test


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
plt.ylabel('RP (V) - GBR Predicted')
plt.title('Test Dataset')

y = clf.predict(X_train)
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
plt.ylabel('RP (V) - GBR Predicted')
plt.title('Train Dataset')

plt.show()

