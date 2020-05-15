import math
import numpy as np
import pandas as pd
from ubknn import *

N_U = 943
N_M = 1682
k = 10

df = pd.read_csv("data.csv")


model = ubknn(df, k)


err_abs_sum = 0
err_sum = 0
err_abs = 0
err = 0
count = 1
for u in range(N_U):
 for m in range(N_M):
  if(not math.isnan(model.getRating(u,m))):
   m_err = model.genRating(u,m) - model.getRating(u,m)
   err_sum += m_err
   err_abs_sum += abs(m_err)
   count += 1
 err_abs = err_abs_sum/count
 err = err_sum/count
 print(u, err, err_abs)






