import pandas as pd

from ubknn import *

df = pd.read_csv("data.csv")


model = ubknn(df, 10)
print(model.score(1,1))
print(model.score(2,2))

