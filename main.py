# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 09:15:34 2020

@author: ghaderi1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_context('talk')
sns.set_style('white')
from sklearn.model_selection import train_test_split

from model import UQ

if __name__ == "__main__": 
    
  
  
    sigma = .07
    
    dataset = pd.read_excel("SBR.xlsx", escapechar="\\")
    X00 = dataset.iloc[:,0:1].values
    Y00 = dataset.iloc[:,1:2].values
    
    X0, x_test, y, y_test = train_test_split(X00, Y00, test_size=0.01)
    
    
    # Caroll Model
    
    X1 = 2*(X0-X0**-2)
    X2 = 8*(2*X0**-1+X0**2)**3*(X0-X0**-2)
    X3 = (1+2*X0**3)**-.5*(X0-X0**-2)
    
    XS1 = np.append(X1,X2,1)
    
    X = np.append(XS1,X3,1)
    
    
    # Define model
    m = UQ(X, y, sigma, alpha=10000, beta=.01)
    
    # Fit MLE and MAP estimates for w
    w_MLE = m.fit_MLE()
    w_MAP, Lambda_inv,kk,S = m.fit_MAP()
    w_MAP = kk
    y_MLE = np.matmul(X, w_MLE)
    y_L = np.sort(np.reshape(y_MLE,(np.size(X0))))
    y_MAP = np.matmul(X, w_MAP)
    y_p = np.sort(np.reshape(y_MAP,(np.size(X0))))
    X_p = np.sort(np.reshape(X0,(np.size(X0))))
    y_MAP_L = y_p - 2*sigma
    y_MAP_U = y_p + 2*sigma
    
    # Predict at a set of test points
    X_star0 = np.linspace(1,4.96,47)[:,None]
    
    X_star1 = 2*(X_star0-X_star0**-2)
    X_star2 = 8*(2*X_star0**-1+X_star0**2)**3*(X_star0-X_star0**-2)
    X_star3 = (1+2*X_star0**3)**-.5*(X_star0-X_star0**-2)
    X_star11 = np.append(X_star1, X_star2,1)
    
    X_star = np.append(X_star11, X_star3, 1)
    y_pred_MLE = np.matmul(X_star, w_MLE)
    y_pred_MAP = np.matmul(X_star, w_MAP)
    
    
    
    # Draw sampes from the predictive posterior
    num_samples = 47
    mean_star, var_star, var_star1 = m.predictive_distribution(X_star)
    samples = np.random.multivariate_normal(mean_star.flatten(), var_star, num_samples)
    
    # Plot
    plt.figure(1, figsize=(10,8), dpi=80, facecolor='w', edgecolor='k')
    
    
    plt.plot(X_star0, y_pred_MAP, 'k--', linewidth=3.0, label = 'weights sampled from posterior')
    for i in range(0, num_samples):
        plt.plot(X_star0, samples[i,:], 'g', linewidth=0.1)
    plt.plot(X0,y,'bo', markersize = 12, alpha = 0.5, label = "Observation")
    plt.legend(frameon=False,loc='upper left')
    plt.xlabel('$Stretch$')
    plt.ylabel('$Stress$')
    plt.axis([1,4,0,6])
    plt.savefig("1.svg")
    
    plt.figure(2, figsize=(10,8), dpi=80, facecolor='w', edgecolor='k')
    
    plt.plot(X_p, y_p,"--", linewidth=3.0, label = 'MPE Prediction',color='orange', alpha=1)
    plt.plot(X_p, y_L, ":", linewidth=3.0, label = 'MLE Prediction', color='black', alpha=1)
    plt.fill_between(X_p, y_MAP_L, y_MAP_U,facecolor='orange', alpha=.5,label= "Two standard deviations")
    plt.plot(X0,y,'bo', markersize = 12, alpha = 0.5, label = "Observation")
    plt.legend(frameon=False,loc='upper left')
    plt.xlabel('$Stretch$')
    plt.ylabel('$Stress$')
    plt.axis([1,4,0,6])
    plt.savefig("2.svg")
    


