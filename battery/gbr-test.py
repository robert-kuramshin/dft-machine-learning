#doing some testing on the dft calculation database without collecting extra features. 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

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

selector = VarianceThreshold(0.05)
selector.fit_transform(X_train)

print(X_train.shape)
# X_train = selector.fit_transform(X_train)
# X_test = selector.transform(X_test)
print(X_train.shape)

scaler = StandardScaler()  

scaler.fit(X_train)  

X_train = scaler.transform(X_train)  

X_test = scaler.transform(X_test)  


# pca = PCA(15)
# pca.fit_transform(X_train)

# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)


# tuned_parameters = [{'alpha': [0.0001,0.001,0.01,0.1,0.2,0.5],"max_iter":[10000000]}]
# las = GridSearchCV(linear_model.Lasso(), tuned_parameters, cv=10)
# las.fit(X_train, y_train)

# mse = mean_squared_error(y_test, las.predict(X_test))
# print("MSE: %.4f" % mse)

# print(las.best_estimator_.coef_)
# print(las.best_estimator_.intercept_)  

# new_features = []
# count = 0
# for coef in las.best_estimator_.coef_:
#     if(not coef == 0.0):
#         new_features.append(feature_cols[count])
#     count+=1

# print (new_features)

# #splitting into dependant and independant variables
# X_train = train.loc[:, new_features]
# y_train = train["Band gap [eV]"]

# X_test = test.loc[:, new_features]
# y_test = test["Band gap [eV]"]

# scaler = StandardScaler()  

# scaler.fit(X_train)  

# X_train = scaler.transform(X_train)  

# X_test = scaler.transform(X_test)  

#creating regressor and fitting data
#params = {'n_estimators': 500, 'loss': 'ls', 'learning_rate': 0.1, 'min_samples_leaf': 3, 'min_samples_split': 4, 'max_depth': 4}

tuned_parameters = [{'n_estimators': [1500,5000],
                     'max_depth': [1,2,3,4,5,6,7,8],
                     'min_samples_leaf':[1,2,3,4,5,6,7],
                    'min_samples_split': [0.5,0.75,1.0,2,3,4],
                    'learning_rate': [0.05,0.075,0.1,0.125],
                'loss': ['ls','lad','huber']}]

scores = ['neg_mean_squared_error']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(GradientBoostingRegressor(), tuned_parameters, cv=3, n_jobs=4,
                       scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()


params = clf.best_params_
reg = GradientBoostingRegressor(**params)

reg.fit(X_train, y_train)
mse = mean_squared_error(y_test, reg.predict(X_test))
print("MSE: %.4f" % mse)
print(r2_score(y_test, reg.predict(X_test)))

# #############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

#write file
output = pd.DataFrame(y_test).copy()
output["Predicted"] = pd.DataFrame(y_pred)
output.to_csv("results.csv")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# ############################################################################
# Plot feature importance
feature_importance = reg.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
feature_names = [feature_names[i] for i in sorted_idx]
plt.yticks(pos, feature_names)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
