 import math
import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

N_U = 943
N_M = 1682
FILE_NAME = 'data.csv'

N_EPOCHS = 10
ALPHA = 0.01
SVD_K = 5

k = 10



def ubknn_algorithm(R):
	return(ubknn(R, k).getPredictionMatrix())
	
def svd_algorithm(R):
	return(SVD(R, k = SVD_K, n_epochs = N_EPOCHS, alpha = ALPHA).getPredictionMatrix())
  

 
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
 
 