import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

#read data
test = pd.read_csv("../data/test.csv")
train = pd.read_csv("../data/train.csv")


#specify feature column names
feature_cols = [
'Radius A [ang]',
'Radius B [ang]',
'Formation energy [eV/atom]',
'Stability [eV/atom]',
'Magnetic moment [mu_B]',
'Volume per atom [A^3/atom]',
'a [ang]',
'b [ang]',
'c [ang]',
'alpha [deg]',
'beta [deg]',
'gamma [deg]',
'Vacancy energy [eV/O atom]',
'Octahedral Factor',
'Tolerance Factor',
'A Ionization Energy',
'B Ionization Energy',
'A Electronegativity',
'B Electronegativity',
"A rs",
"A rp",
"A rd",
"B rs",
"B rp",
"B rd",
"B EA",
"B s total",
"B p total",
"B d total",
"B f total",
"A g",
"A p",
"B g",
"B p"
]


feature_names = [
'Radius A [ang]',
'Radius B [ang]',
'Formation energy [eV/atom]',
'Stability [eV/atom]',
'Magnetic moment [mu_B]',
'Volume per atom [A^3/atom]',
'a [ang]',
'b [ang]',
'c [ang]',
'alpha [deg]',
'beta [deg]',
'gamma [deg]',
'Vacancy energy [eV/O atom]',
'Octahedral Factor',
'Tolerance Factor',
'A Ionization Energy',
'B Ionization Energy',
'A Electronegativity',
'B Electronegativity',
'Goldschmidt Tolerance Factor'
]

#splitting into dependant and independant variables
X_train = train.loc[:, feature_cols]
y_train = train["Band gap [eV]"]

X_test = test.loc[:, feature_cols]
y_test = test["Band gap [eV]"]
tuned_parameters = [{'n_estimators': [1500,5000],
                     'max_depth': [1,2,3,4,5,6,7,8],
                     'min_samples_leaf':[1,2,3,4,5,6,7],
                    'min_samples_split': [0.5,0.75,1.0,2,3,4],
                    'learning_rate': [0.05,0.075,0.1,0.125],
                'loss': ['ls','lad','huber']}]

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

opt = BayesSearchCV( GradientBoostingRegressor(), { 'n_estimators': Integer(1000, 10000),'max_depth': Integer(1, 10),'min_samples_split': Real(0.1, 1, prior='log-uniform'), 'learning_rate': Real(0.1, 1, prior='log-uniform'), 'min_samples_leaf': Integer(1,10), 'loss': Categorical(['ls','lad','huber']), }, n_iter=200,cv=3,n_jobs=4 )

opt.fit(X_train, y_train)

print(opt.score(X_test, y_test))
