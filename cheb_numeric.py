#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 17:45:10 2024

@author: jitongd
"""

import numpy as np


# get the first-order differential matrix 'D' and the Chebyshev distribution grid 'x'
def cheb(N):
    if N == 0:
        D = np.array([0])
        x = np.array([1])
        return D, x
    
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2
    c[-1] = 2
    c = c * (-1) ** np.arange(N + 1)
    
    X = np.tile(x, (N + 1, 1))
    X = X.T
    dX = X - X.T
    
    D = (np.outer(c, 1 / c)) / (dX + np.eye(N + 1))  # off-diagonal entries
    D = D - np.diag(np.sum(D, axis=1))               # diagonal entries
    
    return D, x

# get the Chebyshev integration weight vector 'w'
def clenCurt(N):
    theta = np.pi * np.arange(N + 1) / N
    x = np.cos(theta)
    
    w = np.zeros(N + 1)
    ii = np.arange(1, N)
    v = np.ones(N - 1)
    
    if N % 2 == 0:
        w[0] = 1 / (N**2 - 1)
        w[N] = w[0]
        for k in np.arange(1, N // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
        v -= np.cos(N * theta[ii]) / (N**2 - 1)
    else:
        w[0] = 1 / N**2
        w[N] = w[0]
        for k in range(1, (N + 1) // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
    
    w[ii] = 2 * v / N
    
    return x, w




