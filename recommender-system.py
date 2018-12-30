"""*********************************************************

NAME:     Recommender System

AUTHOR:   Paul Haddon Sanders IV, Ph.D

DATE:     12/30/2018

*********************************************************"""

import numpy as np
from scipy import io
import pandas as pd
from pdb import set_trace as pb

### Get data ###############################################

mat1 = io.loadmat('ex8_movies.mat')

Y = mat1['Y']
R = mat1['R']

############################################################
###         CLASS : COLLABORATIVE FILTERING              ###
############################################################

class collaborative_filtering:
    """Collaborative filtering cost function.
    Parameters
    ----------
    n_features : int
      Number of features in describing the movie.
    eta : float
      Learing rate (between 0.0 and 1.0).
    lambd : float
      Regularization parameter (between 0.0 and 100.0).
    n_iter : int
      Number of passes over the training set.
    
    Attributes
    ----------
    x_ : 2d-array, shape = [n_movies,n_features]
      Attributes for each movie, where n_movies is the number of movies, and n_features is the number of features.
    w_ : 2d-array, shape = [n_users,n_features]
      Attributes for each movie, where n_users is the number of users, and n_features is the number of features.
    y_ : 2d-array, shape = [n_movies,n_users]
      Ratings for each user on each movie, where n_movies is the number of movies, and n_users is the number of features.
    r_ : 2d-array, shape = [n_movies,n_users]
      Identifier of whether a user has rated a movie or not; 1 if rated and 0 if not rated.
    cost_ : list
      Linear regression cost function value in each epoc.

    """
    
    def __init__(self,n_features=10,eta=0.005,lambd=2.0,n_iter=400):

        self.n_features = n_features
        self.eta        = eta
        self.lambd      = lambd
        self.n_iter     = n_iter
        
    def fit(self,Y,R):
        """Fit training data.
        
        Y : 2d-array, shape = [n_movies,n_users]
          Ratings for each user on each movie, where n_movies is the number of movies, and n_users is the number of features.
        R : 2d-array, shape = [n_movies,n_users]
          Identifier of whether a user has rated a movie or not; 1 if rated and 0 if not rated.

        Returns
        -------
        self : object

        """

        self.cost_ = []

        for i in range(self.n_iter):

            print('   {}% complete.'.format(round((i+1)/self.n_iter*100,2)),end='\r')

            dx_,dw_ = self.dcost(self.x_,self.w_,Y,R,self.lambd)
            self.x_ += self.eta * dx_
            self.w_ += self.eta * dw_
            self.cost_.append(self.cost(self.x_,self.w_,Y,R,self.lambd))
        
    def cost(self,X,T,Y,R,lambd):
        """Compute cost"""
        
        J = 0.5 * ((X.dot(T.T) - Y)**2)[R == 1].sum() +0.5 * lambd * ((T**2).sum() + (X**2).sum())
        
        return J

    def dcost(self,X,T,Y,R,lambd):
        """Compute cost gradient"""
        
        dx_ = ((Y - X.dot(T.T)) * R)   .dot(T) - lambd * X
        dw_ = ((Y - X.dot(T.T)) * R).T .dot(X) - lambd * T
        
        return dx_,dw_

    def activation(self,xi,wj):
        """Compute linear activation"""

        z = np.dot(xi,wj)
        
        return z
    
    def predict(self,xi,wj):
        """Return ratings prediction"""

        pred = self.activation(xi,wj)   

        return pred
    
    def add_review(self,file,Y,R):
        """Upload a user-defined review"""

        n_movies = Y.shape[0]
        n_users  = Y.shape[1]
        
        mov_dat = pd.read_csv(file)
        
        self.user = mov_dat.columns[-1]
        
        Y_add = np.zeros((Y.shape[0],len(mov_dat.columns[3:])))
        R_add = np.zeros(Y_add.shape)

        R_val = mov_dat.iloc[:,3:].notnull()

        for user in range(R_val.shape[1]):
    
            Y_num = mov_dat['Number'][R_val.iloc[:,user]] - 1
            Y_val = mov_dat.iloc[:,3+user][R_val.iloc[:,user]]
            Y_add[Y_num,user] = Y_val
            R_add[Y_num,user] = 1
        
        Y = np.append(Y,Y_add,axis=1)
        R = np.append(R,R_add,axis=1)

        Ymean = Y.sum(1) / R.sum(1)
        
        Y = (Y - Ymean.reshape(Y.shape[0],1)) * R
        
        self.x_ = np.random.normal(loc=0.0,scale=1.0,size=(n_movies,self.n_features))
        self.w_ = np.random.normal(loc=0.0,scale=1.0,size=(n_users+1,self.n_features))
        
        self.fit(Y,R)

        self.recommend_val = self.x_.dot(self.w_[-1]) + Ymean
        self.recommend_mov = self.recommend_val.argsort()[::-1]
        self.recommend_val = np.round(np.sort(self.recommend_val)[::-1],3)
        self.recommend_mov = [list(mov_dat['Movie'][mov_dat['Number'] == rec + 1].values) for rec in self.recommend_mov]
        self.recommend_mov = [item for sublist in self.recommend_mov for item in sublist]

############################################################
#                                                          #
############################################################

print('')
print('   COLLABORATIVE FILTERNING COST ALGORITHM')
print('   CREATED BY: PAUL SANDERS')

print('')
print('   Learning the initial parameters...')
print('')

cf = collaborative_filtering()
cf.add_review('movie_ratings.csv',Y,R)

print('   Considering your reviews...')
print('')
print('\n   Based on our thorough analysis {}...'.format(cf.user))
print('   We suggest you watch :')

for i,mov in enumerate(cf.recommend_mov[:20]):
    print('   Predicted rating {} for movie {}'.format(cf.recommend_val[i],mov))

############################################################
