#gbr based recursive feature elimination results
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

in_y_pred = pd.read_csv("res/gbr_raw.csv")
in_y_real = pd.read_csv("res/y_test.csv",index_col=0)

print (mean_squared_error(in_y_real, in_y_pred))
print (r2_score(in_y_real, in_y_pred))