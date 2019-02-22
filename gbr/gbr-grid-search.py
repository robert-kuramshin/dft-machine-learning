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
csv_data = pd.read_csv("data/processed.csv")

#random shuffle
csv_data.sample(frac=1)

#specify feature column names
feature_cols = [
"Radius A [ang]",
"Radius B [ang]",
"Formation energy [eV/atom]",
"Volume per atom [A^3/atom]",
"Goldschmidt Tolerance Factor",
"A Electronegativity",
"B Electronegativity",
"C Electronegativity",
"A Ionization Energy",
"B Ionization Energy",
"C Ionization Energy",
]

X = csv_data.loc[:, feature_cols]
y = csv_data["Band gap [eV]"]

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

tuned_parameters = [{'n_estimators': [250,500,750,1000], 'max_depth': [4,8,12,16], 'min_samples_split': [2,3,4],
          'learning_rate': [0.01,0.05,0.1,0.2], 'loss': ['ls']}]

scores = ['neg_mean_squared_error', 'r2']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(GradientBoostingRegressor(), tuned_parameters, cv=5,
                       scoring='%s' % score)
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

