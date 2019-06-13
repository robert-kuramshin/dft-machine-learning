import numpy as np
import pandas as pd

X_train = pd.read_csv("res/X_train_lasso.csv",index_col=0)
X_test = pd.read_csv("res/X_test_lasso.csv",index_col=0)
y_train = pd.read_csv("res/y_train.csv",index_col=0)
y_test = pd.read_csv("res/y_test.csv",index_col=0)

corr = X_train.corr()
corr.to_csv("res/corr.csv")

# X_test.to_csv("res/X_test_lasso.csv")
# X_train.to_csv("res/X_train_lasso.csv")