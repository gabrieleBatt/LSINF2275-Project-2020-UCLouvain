import pandas as pd
import numpy as np

class SVD():
 def __init__(self, data, k = 10, n_epochs = 30, alpha = 0.01):
  self.n_users = data.shape[0]
  self.n_items = data.shape[1]
  self.R = np.matrix(data)
  self.k = k
  self.n_epochs = n_epochs
  self.alpha = alpha
  self.rating_list = self.__rating_list()
  self.Rp = self.__SGD()
  
 def __rating_list(self):
  ratings = []
  for u in range(self.n_users):
   for m in range(self.n_items):
    if(not np.isnan(self.R[u,m])):
     ratings += [(u,m,self.R[u,m])]
  return(ratings)
 
 
 
 def __SGD(self):
  p = np.random.normal(0, .1, (self.n_users, self.k))
  q = np.random.normal(0, .1, (self.n_items, self.k))
  
  for e in range(self.n_epochs):
   print('\r          \rEpochs:%d/%d' % (e+1, self.n_epochs), end='')
   for u, m, r_ui in self.rating_list:
    err = r_ui - np.dot(p[u], q[m])
    p[u] += self.alpha * err * q[m]
    q[m] += self.alpha * err * p[u]
  
  print('\r            \r', end='')
  return(np.dot(p,q.T))
 
 
 def getRating(self, u, m):
  return(self.R[u,m])
 
 
 def predictedRating(self, u, m):
  return(self.Rp[u,m])






