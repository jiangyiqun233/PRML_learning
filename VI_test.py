#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:44:59 2021

@author: Patrick
"""

import numpy as np

def _init_params(self, X, random_state=None):
    '''
    Method for initializing model parameters based ont teh size and variance of the input data array.

    Parameters
    ----------
    X : 2D numpy array
        2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.
    random_state : int
        int, specifying the seed for random initialization of m


    '''
    N, D =X.shape
    rnd = np.random.RandomState(seed=random_state)
    
    self.alpha = (self.alpha0 + N / self.K) * np.ones(self.K)
    self.beta = (self.beta0 + N / self.K) * np.ones(self.K)
    self.nu = (self.nu0 + N / self.K) * np.ones(self.K)
    self.m = X[rnd.randint(low=0, high=N, size=self.K)]
    self.w = np.tile(np.diag(1.0/np.var(X, axis=0)), (self.K, 1, 1))
    

def _e_like_step(self, X):
    '''
    Method for calculating the array corresponding to responsibility.

    Parameters
    ----------
    X : 2D numpy array
        2D numpy array representing input data, where X[n, i] represents the i-th element of n-th point in X.

    Returns
    -------
    r : 2D numpy array
        2D numpy array representing responsibility of each component for each sample in X,
        where r[n, k] = $r_{n, k}$.

    '''
    N, _ = np.shape(X)
    tpi = np.exp( digamma(self.alpha) - digamma(self.alpha.sum()))
    
    arg_digamma = np.reshape(self.nu, (self.K, 1)) - np.reshape(np.arange(0, self.D, 1), (1, self.D))
    tlam =np.exp(digamma(arg_digamma/2).sum(axis=1))
    