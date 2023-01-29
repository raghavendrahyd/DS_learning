import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


#Note, Lasso has no closed form solution for ommiting Lasso

def reg_loss(X, y, B, lmbda=0):
    """
    Regression loss calculation. Suppy lambda for ridge regression.
    Default lambda is 0 i.e no regularization
    """
    return np.dot(np.transpose(y - np.dot(X, B)),y - np.dot(X, B)) + lmbda*np.dot(np.transpose(B),B)

def loss_gradient(X, y, B, lmbda=0):
    """
    Regression loss function gradient calculation. Suppy lambda for ridge regression.
    Default lambda is 0 i.e no regularization
    """         
    return -np.dot(np.transpose(X), y - np.dot(X, B)) + lmbda*B

def log_likelihood(X, y, B,lmbda=0):
    """
    Log likelihood function calculation. Suppy lambda for ridge regression.
    Default lambda is 0 i.e no regularization
    """         
    logit_res = np.dot(X, B)
    return -np.sum( y*logit_res - np.log(1 + np.exp(logit_res)) ) - lmbda*np.dot(np.transpose(B),B)

def sigmoid(z):
    return (1)/((1+np.exp(-z)))

def log_likelihood_gradient(X, y, B, lmbda=0):
    logit_res = np.dot(X, B)
    preds = sigmoid(logit_res)
    return -np.dot(np.transpose(X),(y-preds)) + lmbda*B

def normalize(X):
    """ 
    normalize x as
    (x-mean)/std_dev
    """
    # When input is dataFrame
    if isinstance(X, pd.DataFrame): 
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                std = np.std(X[c])
                X[c] = (X[c] - u) / std
        return
    #when input is numpy array
    for j in range(X.shape[1]):
        u = np.mean(X[:,j])
        std = np.std(X[:,j])
        X[:,j] = (X[:,j] - u) / std

def minimize_grad_desc(X, y, loss_gradient,
              eta=0.00001, lmbda=0.0,
              max_iter=1000, addB0=True,
              precision=1e-9):
    
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")
    n, p = X.shape
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")

    if addB0: # for Ridge regression, set addB0 to False
        X0 = np.ones((n,1))
        X = np.hstack((X0, X))
        p += 1

    # initiate a random vector of Bs between [-1,1)
    B = np.random.random_sample(size=(p, 1)) * 2 - 1

    prev_B = B
    eps = 1e-5 # To prevent division by 0
    
    h = 0
    for i in range(max_iter):        
        g = loss_gradient(X, y, prev_B, lmbda)
        h = np.add(h, g**2)
        if np.linalg.norm(g,2)<=precision:break #to check stopping condition
        B = np.subtract(prev_B ,np.multiply((eta/np.sqrt(h+eps)) , g))
        prev_B = B
    return B


class LinearRegression:
    def __init__(self,eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize_grad_desc(X, y, loss_gradient, self.eta, self.lmbda, self.max_iter)

class RidgeRegression:
    def __init__(self,eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        normalize(X)
        self.B = np.concatenate(([[np.mean(y)]], \
                    minimize_grad_desc(X, y,
                          loss_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter,addB0=False)),axis=0)

class LogisticRegression: 
    def __init__(self, eta=0.00001, lmbda=0.0, max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict_proba(self, X):
        """
        Computes the probability that the target is 1
        """
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        logits_results= np.dot(X, self.B)
        return sigmoid(logits_results)

    def predict(self, X, threshold = 0.5):
        """
        Computes prediction with respect to the given threshold
        """
        return np.where(self.predict_proba(X)>threshold,1,0)

    def fit(self, X, y):
        self.B = minimize_grad_desc(X, y,log_likelihood_gradient,self.eta,self.lmbda,self.max_iter)
