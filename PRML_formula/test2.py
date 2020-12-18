#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:46:32 2020

@author: Patrick
"""

import numpy as np

X = np.array([[1,2,3],[3,2,1]])

print(X.shape)

w_cov = np.array([[1,1,1],[1,1,1],[1,1,1]])

result = X @ w_cov * X


a = X @ w_cov 
print(a)

print(result.shape,result)

s = np.sum(result, axis=1)

print(s)

other = np.sum(X @ w_cov @ X.T, axis=1)
print(other, other.shape)