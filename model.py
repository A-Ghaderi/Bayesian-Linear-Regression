# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 09:15:00 2020

@author: ghaderi1
"""

import numpy as np
import scipy as scipy

class UQ:
  
  def __init__(self, X, y, sigma = .025 , alpha = 1000, beta = .01):
      
      self.X = X
      self.y = y
      self.sigma = sigma
      self.alpha = alpha
      self.beta = beta
      self.jitter = 1e-8
      

  def fit_MLE(self): 
      xTx_inv = np.linalg.inv(np.dot(self.X.T,self.X) + self.jitter)
      xTy = np.dot(self.X.T, self.y)
      w_MLE = np.dot(xTx_inv, xTy)
      
      self.w_MLE = w_MLE
      
      return w_MLE
      
  def fit_MAP(self): 
      Lambda = (np.dot(self.X.T,self.X)) / self.sigma**2 +  (self.beta/self.alpha)*np.eye(self.X.shape[1])
      Lambda_inv = np.linalg.inv(Lambda)
      xTy = np.dot(self.X.T, self.y)
      mu = np.dot(Lambda_inv, xTy)
      
      L = scipy.linalg.cho_factor(Lambda)
      mu1 = scipy.linalg.cho_solve(L, np.dot(self.X.T, self.y) / self.sigma**2)
      S = scipy.linalg.cho_solve(L, np.eye(self.X.shape[1]))
      
      self.L = L
      self.S = S
      self.w_MAP1 = mu1
      self.w_MAP = mu
      self.Lambda_inv = Lambda_inv
      
      return mu, Lambda_inv, mu1,S
      
  def predictive_distribution(self, X_star):
      mean_star = np.dot(X_star, self.w_MAP1)
      var_star = 1.0/self.alpha + \
                 np.dot(X_star, np.dot(self.Lambda_inv, X_star.T))
                 
      var_star1 = 1.0/self.alpha + \
                 np.dot(X_star, np.dot(self.Lambda_inv, X_star.T))/self.sigma**2        
      return mean_star, var_star, var_star1