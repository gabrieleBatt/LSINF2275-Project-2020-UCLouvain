import math
import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

N_FOLDS = 10
N_EPOCHS = 5
ALPHA = 0.01
SVD_K = 10


N_U = 943
N_M = 1682
k = 10

def main(): 
	
	data = readData()
	
	folds = createFolds(data)
	
	print('UBKNN')
	
	avg_ubknn_MSE, avg_ubknn_MAE = crossValidation(data, folds, method='ubknn')
	
	print('Average SVD MSE:\t%f' % avg_ubknn_MSE)
	print('Average SVD MAE:\t%f' % avg_ubknn_MAE)
	
	
	print('SVD')
	
	avg_svd_MSE, avg_svd_MAE = crossValidation(data, folds, method='svd')
	
	print('Average SVD MSE:\t%f' % avg_svd_MSE)
	print('Average SVD MAE:\t%f' % avg_svd_MAE)
	
	
	
	
	
	


def readData():
	print('Reading data...')
	return(np.matrix(pd.read_csv("data.csv")))

def createFolds(data): 
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
	
	folds += [available_ratings[(N_FOLDS-1)*fold_size:]]
	
	return(folds)

def crossValidation(data, folds,method='svd'):
	print('Training and validation...')
	avg_MSE = 0
	avg_MAE = 0
	for f, fold in enumerate(folds):
		print('Fold:%d/%d' % (f+1, N_FOLDS))
		MSE, MAE = validate(data, fold, method = method)
		print('MSE:\t%f' % MSE)
		print('MAE:\t%f' % MAE)
		avg_MSE += MSE
		avg_MAE += MAE
	
	avg_MSE /= N_FOLDS
	avg_MAE /= N_FOLDS
	return(avg_MSE,avg_MAE)

 
def validate(data, fold, method='svd'):
	partial_data = data.copy()
	for u,m in fold:
		partial_data[u,m] = np.nan

	if(method == 'svd'):
		ratingAlgo = SVD(partial_data, k = SVD_K, n_epochs = N_EPOCHS, alpha = ALPHA)
	elif(method == 'ubknn'):
		ratingAlgo = ubknn(partial_data, k = k)
		
	
	MSE = 0
	MAE = 0
	for i,(u,m) in enumerate(fold):
		print('\r                   \rValidating:%.2f%%' % (i/len(fold)), end='')
		rp = ratingAlgo.predictedRating(u,m)
		MSE += (data[u,m] - rp)**2
		MAE += abs(data[u,m] - rp)
	
	print('\r                   \r')
	MSE = MSE/len(fold)
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
		self.R_hat = self.__SGD()
  
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
 
	def getPredictionMatrix():
		return(self.R_hat)
 
	def getRating(self, u, m):
		return(self.R[u,m])
 
 
	def predictedRating(self, u, m):
		return(self.R_hat[u,m])


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
		usersWhoVoted = self.S.iloc[self.Rsparse[m].dropna().index,:]
		if(usersWhoVoted.shape[0] == 0):
			usersWhoVoted = self.S
		return(list(usersWhoVoted.sort_values(u, ascending = False).head(k).index))
 
 
 
	def __generateScore(self, u, m, verbose = False):
		neighbours = self.__kNNs(u, self.k, m)
		
		similarities = np.array(self.S.iloc[u, neighbours])
		rel_rating = np.array(self.R.iloc[neighbours, m]) - np.array(self.__averageScore(neighbours))
		
		score = self.__averageScore(u) + np.sum(rel_rating*similarities)/np.sum(similarities)
		
		if(verbose):
			print("S:\t\t", similarities)
			print("Ratings:\t", np.array(self.R.iloc[neighbours, m]))
			print("Avgs:\t\t", np.array(self.__averageScore(neighbours)))
			print("Relative:\t", rel_rating)
			print("Avg score:\t", self.__averageScore(u))
			print("Score:\t\t", score)
			
		return(score)
 
 
 
	def __averageScore(self, u):
		return(self.R.iloc[:,u].mean(axis=0))
  
  
	def getPredictionMatrix():
		self.R_hat = np.zeros(self.R.shape)
		for u in range(N_U):
			print('\r                              \rGenerating predictions:%.2f%%' % (u/N_U), end='')
			for m in range(N_M):
				self.R_hat = self.__generateScore(u,m)
		print('\r                              \r', end='')
		return(self.R_hat)
 
 
	def getRating(self, u, m):
		return(self.Rsparse.iloc[u,m])
 
 
 
	def predictedRating(self, u, m):
		return(self.__generateScore(u,m))
 



 
if __name__ == '__main__':
    main()

	
	

