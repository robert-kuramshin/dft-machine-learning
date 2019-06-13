import numpy as np
import pandas as pd

from itertools import combinations

test = pd.read_csv("res/X_test_trans.csv",index_col=0)
train = pd.read_csv("res/X_train_trans.csv",index_col=0)

cc = list(combinations(test.columns,2))
test2 = pd.concat([test[c[1]].mul(test[c[0]]) for c in cc], axis=1, keys=cc)
test2.columns = test2.columns.map(' x '.join)
ccc = list(combinations(test.columns,3))
test3 = pd.concat([test[cc[2]].mul(test[cc[1]]).mul(test[cc[0]]) for cc in ccc], axis=1, keys=ccc)
test3.columns = test3.columns.map(' x '.join)

X_test = pd.concat([test,test2,test3], axis=1)
print(X_test)

cc = list(combinations(train.columns,2))
train2 = pd.concat([train[c[1]].mul(train[c[0]]) for c in cc], axis=1, keys=cc)
train2.columns = train2.columns.map(' x '.join)
ccc = list(combinations(train.columns,3))
train3 = pd.concat([train[cc[2]].mul(train[cc[1]]).mul(train[cc[0]]) for cc in ccc], axis=1, keys=ccc)
train3.columns = train3.columns.map(' x '.join)

X_train = pd.concat([train,train2,train3], axis=1)
print(X_train)

X_test.to_csv("res/X_test_comp.csv")
X_train.to_csv("res/X_train_comp.csv")

