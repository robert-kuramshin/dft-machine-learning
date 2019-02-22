print(__doc__)

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

# Build a classification task using 3 informative features
#read data
csv_data = pd.read_csv("data/dft.csv")

#random shuffle
csv_data.sample(frac=1)

#specify feature column names
feature_cols = [
"Radius A [ang]",
"Radius B [ang]",
"Formation energy [eV/atom]",
"Stability [eV/atom]",
"Magnetic moment [mu_B]",
"Volume per atom [A^3/atom]",
"a [ang]",
"b [ang]",
"c [ang]",
"alpha [deg]",
"beta [deg]",
"gamma [deg]",
"Vacancy energy [eV/O atom]"
]

#remove rows with missing feature values
for feature in feature_cols:
    csv_data = csv_data[csv_data[feature] != "-"]

csv_data = csv_data[csv_data["Band gap [eV]"] != "0.000"]

#splitting into dependant and independant variables
X = csv_data.loc[:, feature_cols]
y = np.array(csv_data["Band gap [eV]"].values).astype(float)

# Create the RFE object and compute a cross-validated score.
params = {'n_estimators': 750, 'max_depth': 4, 'learning_rate': 0.05, 'loss': 'ls', 'min_samples_split': 2}
gbr = GradientBoostingRegressor(**params)
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=gbr, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()