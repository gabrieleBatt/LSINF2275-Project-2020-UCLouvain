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
  return(df.apply(lambda row: row.fillna(row.mean()), axis=1))
 
 
 
 def __similarityMatrix(self, df):
  cs = cosine_similarity(df)
  np.fill_diagonal(cs, 0)
  return(pd.DataFrame(cs,index = df.index))
 
 
 
 def __kNNs(self, u, k, m):
  usersWhoVoted = self.S.iloc[:,self.Rsparse['%d' % m].dropna().index]
  if(usersWhoVoted.shape[0] == 0):
   usersWhoVoted = self.S
  return(list(usersWhoVoted.sort_values([u], ascending = False).head(k).index))
 
 
 
 def __generateScore(self, u, m):
  neighbours = self.__kNNs(u, self.k, m)
  
  similarities = np.array(self.S.iloc[u, neighbours])
  rel_rating = np.array(self.R.iloc[neighbours, m]) - np.array(self.__averageScore(neighbours))
  
  print("S:\t\t", similarities)
  print("Ratings:\t", np.array(self.R.iloc[neighbours, m]))
  print("Avgs:\t\t", np.array(self.__averageScore(neighbours)))
  print("Relative:\t", rel_rating)
  
  score = self.__averageScore(u) + np.sum(rel_rating*similarities)/np.sum(similarities)
  
  print("Avg score:\t", self.__averageScore(u))
  print("Score:\t\t", score)
  
  return(score)
 
 
 
 def __averageScore(self, u):
  return(self.R.iloc[:,u].mean(axis=0))
  
 
 def getRating(self, u, m):
  return(self.Rsparse.iloc[u,m])
 
 
 
 def genRating(self, u, m):
  return(self.__generateScore(u, m))
 
 
 def setK(k):
  self.k = k

