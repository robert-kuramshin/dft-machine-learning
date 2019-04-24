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

##################### 1. READ DATA #####################
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
"A s orbital radius",
"A p orbital radius",
"A d orbital radius",
"B s orbital radius",
"B p orbital radius",
"B d orbital radius",
"B Electron Affinity",
"B s orbital atoms",
"B p orbital atoms",
"B d orbital atoms",
"B f orbital atoms",
"A group",
"A peroid",
"B group",
"B period"
]

##################### 2. SPLIT INTO TEST AND TRAIN #####################
print("Splitting")
X_train = train.loc[:, feature_cols]
y_train = train["Band gap [eV]"]

X_test = test.loc[:, feature_cols]
y_test = test["Band gap [eV]"]

##################### 3. NORMALIZATION #####################
print("Norm")
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

##################### 4.1 TRANSFORMED FEATURES ################
#transforms
print("Transforms")
transforms = [lambda x: np.power(x,2),lambda x: np.power(x,3),lambda x: np.power(x,1/2),lambda x: np.log(1+x),lambda x: np.exp(x)]
n_cols = len(X_train.columns)
for col in range(n_cols):
    for trans in transforms:
        X_train[len(X_train.columns)] = 0 # add column
        X_train[len(X_train.columns)-1] = X_train.iloc[:,col]
        X_train.iloc[:,len(X_train.columns)-1].transform(trans)

        X_test[len(X_test.columns)] = 0 # add column
        X_test[len(X_test.columns)-1] = X_test.iloc[:,col]
        X_test.iloc[:,len(X_test.columns)-1].transform(trans)
#x, x1/2, ln(1 + x), and ex 


n_cols = len(X_train.columns)
##################### 4.2 COMPOSITE FEATURES ################

#compounding
count = 0
print("Compounding Part 1")
for f_1 in range(n_cols):
    for f_2 in range(n_cols):
        print(count," of ",n_cols * (n_cols-1))
        if (not f_1 == f_2):
            X_train[len(X_train.columns)] = 0 # add column
            X_train[len(X_train.columns)-1] = X_train.iloc[:,f_1] * X_train.iloc[:,f_2]
            X_test[len(X_test.columns)] = 0 # add column
            X_test[len(X_test.columns)-1] = X_test.iloc[:,f_1] * X_test.iloc[:,f_2]
            count+=1

#compounding
# count = 0
# print("Part 2")
# for f_1 in range(n_cols):
#     for f_2 in range(n_cols):
#         for f_3 in range(n_cols):
#             print(count," of ",n_cols * n_cols * n_cols)
#             if (not f_1 == f_2 and not f_2 == f_3 and not f_1 == f_3):
#                 X_train[len(X_train.columns)] = 0 # add column
#                 X_train[len(X_train.columns)-1] = X_train.iloc[:,f_1] * X_train.iloc[:,f_2] * X_train.iloc[:,f_3]
#                 X_test[len(X_test.columns)] = 0 # add column
#                 X_test[len(X_test.columns)-1] = X_test.iloc[:,f_1] * X_test.iloc[:,f_2] * X_test.iloc[:,f_3]
#                 count+=1

##################### 4.3 CORRELATION FILTER ################
print("Correlation Filter")

corr_matrix = X_train.corr()

columns = np.full((corr_matrix.shape[0],), True, dtype=bool)
for i in range(corr_matrix.shape[0]):
    for j in range(i+1, corr_matrix.shape[0]):
        if corr_matrix.iloc[i,j] >= 0.999:
            if columns[j]:
                columns[j] = False
selected_columns = X_train.columns[columns]