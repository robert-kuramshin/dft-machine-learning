import numpy as np
import pandas as pd

X_train = pd.read_csv("res/X_train_lasso.csv",index_col=0)
X_test = pd.read_csv("res/X_test_lasso.csv",index_col=0)
y_train = pd.read_csv("res/y_train.csv",index_col=0)
y_test = pd.read_csv("res/y_test.csv",index_col=0)

corr = X_train.corr()
corr.to_csv("res/corr.csv")

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.999:
            if columns[j]:
                columns[j] = False
selected_columns = X_train.columns[columns]

X_test[selected_columns].to_csv("res/X_test_corr.csv")
X_train[selected_columns].to_csv("res/X_train_corr.csv")