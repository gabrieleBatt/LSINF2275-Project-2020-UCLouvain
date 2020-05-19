import math
import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

N_FOLDS = 10
N_U = 943
N_M = 1682
FILE_NAME = 'data.csv'

#SVD params
N_EPOCHS = 10
ALPHA = 0.01
SVD_K = 5

#ubknn params
k = 5

#################################################################################
#################################################################################
#################################################################################

#Read data and perform validation
def main(): 
	
	#preliminaries
	data = readData()
	folds = createFolds(data)
	
	#BASELINE validation
	print('BASELINE')
	
	avg_baseline_MSE, avg_baseline_MAE = crossValidation(data, folds, method='baseline')
	
	print('Average baseline MSE:\t%f' % avg_baseline_MSE)
	print('Average baseline MAE:\t%f' % avg_baseline_MAE)
	
	#UBKNN validation
	print('UBKNN')
	
	avg_ubknn_MSE, avg_ubknn_MAE = crossValidation(data, folds, method='ubknn')
	
	print('Average UBKNN MSE:\t%f' % avg_ubknn_MSE)
	print('Average UBKNN MAE:\t%f' % avg_ubknn_MAE)
	
	#SVD validation
	print('SVD')
	
	avg_svd_MSE, avg_svd_MAE = crossValidation(data, folds, method='svd')
	
	print('Average SVD MSE:\t%f' % avg_svd_MSE)
	print('Average SVD MAE:\t%f' % avg_svd_MAE)
	
#################################################################################
#################################################################################
#################################################################################

#Read from the csv file the data
def readData():
	print('Reading data...')
	return(np.matrix(pd.read_csv(FILE_NAME)))

#divide rthe data in 10 random folds
def createFolds(data): 
	print('Creating folds...')
	#create a list of available ratings
	available_ratings = []
	for u in range(data.shape[0]):
		for m in range(data.shape[1]):
			if(not np.isnan(data[u,m])):
				available_ratings += [(u,m)]
	#randomize the list
	random.shuffle(available_ratings)
	
	#divide in 10 folds
	folds = []
	fold_size = int(len(available_ratings)/N_FOLDS)
	for i in range(N_FOLDS-1):
		folds += [available_ratings[i*fold_size:(i+1)*fold_size]]
	folds += [available_ratings[(N_FOLDS-1)*fold_size:]]
	
	return(folds)

#takes the data, the list of 10 folds and performs cross validation on one of the three algorithms: baseline, ubknn, SVD
def crossValidation(data, folds,method='baseline'):
	print('Training and validation...')
	#inizialize error
	avg_MSE = 0
	avg_MAE = 0
	#for each fold
	for f, fold in enumerate(folds):
		print('Fold:%d/%d' % (f+1, N_FOLDS))
		#computing MSE and MAE
		MSE, MAE = validate(data, fold, method = method)
		print('MSE:\t%f' % MSE)
		print('MAE:\t%f' % MAE)
		#update error
		avg_MSE += MSE
		avg_MAE += MAE
	
	#average
	avg_MSE /= N_FOLDS
	avg_MAE /= N_FOLDS
	
	return(avg_MSE,avg_MAE)

#performs validation on one fold
def validate(data, fold, method='baseline'):
	#removes the raing in the current fold from the available data
	partial_data = data.copy()
	for u,m in fold:
		#set the data to remove at NaN
		partial_data[u,m] = np.nan

	#algorithm selection
	if(method == 'baseline'):
		ratingAlgo = baselinePrediction(partial_data)
	if(method == 'svd'):
		ratingAlgo = SVD(partial_data, k = SVD_K, n_epochs = N_EPOCHS, alpha = ALPHA)
	elif(method == 'ubknn'):
		ratingAlgo = ubknn(partial_data, k = k)
		
	#inizialize error
	MSE = 0
	MAE = 0
	#for each element in the fold
	for i,(u,m) in enumerate(fold):
		print('\r                   \rValidating: %.2f%%' % (i*100.0/len(fold)), end='')
		#prediction (all three classes have this function)
		rp = ratingAlgo.predictedRating(u,m)
		#update error
		MSE += (data[u,m] - rp)**2
		MAE += abs(data[u,m] - rp)
	
	print('\r                   \r')
	
	#average
	MSE = MSE/len(fold)
	MAE = MAE/len(fold)
	
	return(MSE, MAE)
  
#################################################################################
#################################################################################
#################################################################################
 
#This class encloses the SVD algorithm
class SVD():
	#uses as input the training data and the parameters
	def __init__(self, data, k = 10, n_epochs = 30, alpha = 0.01):
		#saving dimension of data
		self.n_users = data.shape[0]
		self.n_items = data.shape[1]
		#coping data into a matrix
		self.R = np.matrix(data)
		#saving parameters
		self.k = k
		self.n_epochs = n_epochs
		self.alpha = alpha
		#creating a list of all  available ratings
		self.rating_list = self.__rating_list()
		#producing prediction matrix
		self.R_hat = self.__SGD()
  
	#converts the training data in a list of available ratings
	#[(user, item, rating), (user, item, rating), ...]
	def __rating_list(self):
		ratings = []
		for u in range(self.n_users):
			for m in range(self.n_items):
				if(not np.isnan(self.R[u,m])):
					ratings += [(u,m,self.R[u,m])]
		return(ratings)
	
	#return the prediction matrix; performs the optimization
	def __SGD(self):
		#initialize randomly p and q (M has p[u] as rows, U^T has q[m] as columns)
		p = np.random.normal(0, .1, (self.n_users, self.k))
		q = np.random.normal(0, .1, (self.n_items, self.k))
  
		#for each epoch we perform one step of gradient descent
		for e in range(self.n_epochs):
			print('\r          \rEpochs:%d/%d' % (e+1, self.n_epochs), end='')
			for u, m, r_ui in self.rating_list:
				err = r_ui - np.dot(p[u], q[m])
				p[u] += self.alpha * err * q[m]
				q[m] += self.alpha * err * p[u]
  
		print('\r            \r', end='')
		
		return
		return(np.dot(p,q.T))
 
	#returns the prediction matrix
	def getPredictionMatrix():
		return(self.R_hat)

	#returns the prediction on une item of one user
	def predictedRating(self, u, m):
		return(self.R_hat[u,m])

	
#################################################################################
#################################################################################
#################################################################################
		
#This class encloses the User based knn algorithm
class ubknn():
	#uses as input the training data and the number of neighbours to consider
	def __init__(self, RIn, k):
		self.k = k
		
		#saving the sparse input matrix
		self.Rsparse = pd.DataFrame(RIn)
		
		#save a version with the missing values filled
		self.R = self.__fillNaN(self.Rsparse)
		
		#computing the similarity between each pair of users
		self.S = self.__similarityMatrix(self.Rsparse)
		
 
	#fill with average rating of item, if the item is never rated, fill witth average rating of the user
	def __fillNaN(self, df):
		return(df.apply(lambda column: column.fillna(column.mean()), axis=0).apply(lambda row: row.fillna(row.mean()), axis=1))
 
 
	#return the similarity matrix
	def __similarityMatrix(self, df):
		df = df.fillna(0)
		cs = cosine_similarity(df)
		#the similarity with oneself is set at 0 to ignore the same user when looking for the neighbours
		np.fill_diagonal(cs, 0)
		return(pd.DataFrame(cs,index = df.index))
 
 
	#selects the best k neighbours among the ones who rated the item, if nobody has done so, it selects the best among all
	def __kNNs(self, u, k, m):
		usersWhoVoted = self.S.iloc[self.Rsparse[m].dropna().index,:]
		if(usersWhoVoted.shape[0] == 0):
			usersWhoVoted = self.S
		return(list(usersWhoVoted.sort_values(u, ascending = False).head(k).index))
 
 
    #score computation
	def __generateScore(self, u, m):
		#selecting best neighbours
		neighbours = self.__kNNs(u, self.k, m)
		
		similarities = np.array(self.S.iloc[u, neighbours])
		rel_rating = np.array(self.R.iloc[neighbours, m]) - np.array(self.__averageRating(neighbours))
		
		score = self.__averageRating([u]) + np.sum(rel_rating*similarities)/np.sum(similarities)
			
		return(score)
 
 
	#average score given by user u
	def __averageRating(self, u):
		return(self.R.iloc[u,:].mean(axis=1).values)
  
	#returns the prediction matrix
	def getPredictionMatrix():
		self.R_hat = np.zeros(self.R.shape)
		#for each user and item, it predicts the rating
		for u in range(N_U):
			print('\r                              \rGenerating predictions: %.2f%%' % (u*100.0/N_U), end='')
			for m in range(N_M):
				self.R_hat = self.__generateScore(u,m)
		print('\r                              \r', end='')
		return(self.R_hat)
 
	#returns the prediction on une item of one user
	def predictedRating(self, u, m):
		return(self.__generateScore(u,m))
 
 
#################################################################################
#################################################################################
#################################################################################


#This class encloses the baseline algorithm
class baselinePrediction():
	def __init__(self, R):
		self.R_hat = self.__fillNaN(pd.DataFrame(R))
	
	
	#fill with average rating of item
	def __fillNaN(self, df):
		return(df.apply(lambda column: column.fillna(column.mean()), axis=0).apply(lambda row: row.fillna(row.mean()), axis=1))

	#returns the prediction matrix
	def getPredictionMatrix():
		return(self.R_hat)
 
	#returns the prediction on une item of one user
	def predictedRating(self, u, m):
		return(self.R_hat.iloc[u,m])

		
	
#################################################################################
#################################################################################
#################################################################################
 
if __name__ == '__main__':
    main()

	
	

