import math
import random
import pandas as pd
import numpy as np


N_FOLDS = 10
N_EPOCHS = 30
ALPHA = 0.01
K = 10

def main(): 
 
 data = readData()
 
 folds = svdCreateFolds(data)
 
 avg_MSE,avg_MAE = svdCrossValidation(data, folds)
 
 print('Average MSE:\t%f' % avg_MSE)
 print('Average MAE:\t%f' % avg_MAE)


def readData():
 print('Reading data...')
 return(np.matrix(pd.read_csv("data.csv")))

def svdCreateFolds(data): 
 print('Creating folds...')
 available_ratings = []
 for u in range(data.shape[0]):
  for m in range(data.shape[1]):
   if(not np.isnan(data[u,m])):
    available_ratings += [(u,m)]
 
 random.shuffle(available_ratings)
 folds = []
 fold_size = int(len(available_ratings)/N_FOLDS)
 for i in range(N_FOLDS-1):
  folds += [available_ratings[i*fold_size:(i+1)*fold_size]]
 
 folds += [available_ratings[9*fold_size:]]
 
 return(folds)

def svdCrossValidation(data, folds):
 print('Training and validation...')
 avg_MSE = 0
 avg_MAE = 0
 for f, fold in enumerate(folds):
  print('Fold:%d/%d' % (f+1, N_FOLDS))
  MSE, MAE = svdValidate(data, fold)
  print('MSE:\t%f' % MSE)
  print('MAE:\t%f' % MAE)
  avg_MSE += MSE
  avg_MAE += MAE
  
 avg_MSE /= N_FOLDS
 avg_MAE /= N_FOLDS
 return(avg_MSE,avg_MAE)

 
def svdValidate(data, fold):
 partial_data = data.copy()
 for u,m in fold:
  partial_data[u,m] = np.nan
 svdRating = SVD(partial_data, k = K, n_epochs = N_EPOCHS, alpha = ALPHA)
 
 MSE = 0
 for u,m in fold:
  MSE += (data[u,m] - svdRating.predictedRating(u,m))**2
 
 MSE = MSE/len(fold)
 
 MAE = 0
 for u,m in fold:
  MAE += abs(data[u,m] - svdRating.predictedRating(u,m))
 
 MAE = MAE/len(fold)
 
 return(MSE, MAE)
  

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


 
if __name__ == '__main__':
    main()

	
	

