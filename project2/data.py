import pandas as pd
import numpy as np

f = open("ml-100k/u.data","r") 

lines = f.readlines()

R = np.empty((1000,1700))  
R[:] = np.nan

for i, line in enumerate(lines):
 s = line.split()[:3] 
 R[int(s[0]), int(s[1])] = int(s[2])


df = pd.DataFrame(R)
 
df.to_csv("data.csv", index = False)