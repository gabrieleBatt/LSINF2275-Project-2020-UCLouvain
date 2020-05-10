import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import math

class ubknn():
 def __init__(self, RIn, k):
  self.k = k
  
  self.Rsparse = pd.DataFrame(RIn)
  
  self.R = self.__fillNaN(self.Rsparse)
  
  self.S = self.__similarityMatrix(self.R)
 
 
 
 def __fillNaN(self, df):
  #fill with average rating of user
  return(df.fillna(df.mean(axis=1)).fillna(df.mean(axis=0)).fillna(0))
 
 
 
 def __similarityMatrix(self, df):
  cs = cosine_similarity(df)
  np.fill_diagonal(cs, 0)
  return(pd.DataFrame(cs,index = df.index))
 
 
 
 def __kNNs(self, u, k):
  return(list(self.S.sort_values([u], ascending = False).head(k).index))
 
 
 
 def __generateScore(self, u, m):
  neighbours = self.__kNNs(u, self.k)
  
  similarities = np.array(self.S.iloc[u, neighbours])
  rel_rating = np.array(self.R.iloc[neighbours, m]) - np.array(self.__averageScore(neighbours))
  
  score = self.__averageScore(u) + np.sum(rel_rating*similarities)/np.sum(similarities)
  
  return(score)
 
 
 
 def __averageScore(self, u):
  return(self.R.iloc[:,u].mean(axis=0))
 
 
 
 def score(self, u, m, forceGen=True):
  if((not math.isnan(self.Rsparse.iloc[u,m])) and (not forceGen)):
   return(self.Rsparse.iloc[u,m])
  else:
   return(self.__generateScore(u, m))
 
 
 def setK(k):
  self.k = k

