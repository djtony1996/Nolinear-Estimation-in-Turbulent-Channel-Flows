#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:11:42 2024

@author: jitongd
"""

import numpy as np
from cheb_numeric import cheb

def OSS_operator(kx,ky,nz,Retau,channelRe,z_int,whe_eddyv):
    
    # ------ derivative matrices ------
    nz = 128
    D, z = cheb(nz)
    D1 = D[1:-1, 1:-1]  
    D2 = np.matmul(D, D)
    D2 = D2[1:-1, 1:-1]
    S = np.diag([0] + [1 / (1 - z[i] ** 2) for i in range(1, nz)] + [0])
    D4 = (np.diag(1 - z**2) @ np.linalg.matrix_power(D, 4)
          - 8 * np.diag(z) @ np.linalg.matrix_power(D, 3)
          - 12 * np.matmul(D, D)) @ S
    D4 = D4[1:nz, 1:nz]
    
    # ------ mean flow and eddy viscosity ------
    U = np.diag(channelRe['Up'][1:-1])
    dUdy = np.diag(channelRe['Up_diff1'][1:-1])
    d2Udy2 = np.diag(channelRe['Up_diff2'][1:-1])
    
    if whe_eddyv == 1:
        vT = np.diag(channelRe['vTpd'][1:-1])
        vT_diff1 = np.diag(channelRe['vTp_diff1'][1:-1])
        vT_diff2 = np.diag(channelRe['vTp_diff2'][1:-1])
    else:
        vT = np.eye(nz-1)
        vT_diff1 = np.zeros((nz-1, nz-1))
        vT_diff2 = np.zeros((nz-1, nz-1))
    
    # ------ Laplacian operators ------
    k2 = kx**2 + ky**2
    I  = np.eye(nz-1)
    L  = D2 - k2 * I
    L2 = D4 + k2**2 * I - 2 * k2 * D2
    
    # ------ Orr-Sommerfeld operator ------
    A1 = (-1j * kx * U @ L + 1j * kx * d2Udy2 + vT @ L2 / Retau + 2 * vT_diff1 @ D1 @ L / Retau + vT_diff2 @ (D2 + k2 * I) / Retau)
    Ao = np.linalg.solve(L, A1)
    
    # ------ Squire operator ------
    As = (-1j * kx * U + vT @ L / Retau + vT_diff1 @ D1 / Retau)
    
    # ------ the coupling term ------
    As1 = -1j * ky * dUdy
    
    # ------ state-space matrices ------
    A = np.block([
        [Ao, np.zeros((nz-1, nz-1))],
        [As1, As]
    ])
    
    B = np.block([
        [np.linalg.solve(L, -1j * kx * D1), np.linalg.solve(L, -1j * ky * D1), np.linalg.solve(L, -k2 * I)],
        [1j * ky * I, -1j * kx * I, np.zeros((nz-1, nz-1))]
    ])
    
    C_ori = np.block([
        [1j * kx * D1, -1j * ky * I],
        [1j * ky * D1, 1j * kx * I],
        [k2 * I, np.zeros((nz-1, nz-1))]
    ]) / k2
    
    
    Cint = np.zeros((len(z_int), nz+1))
    bary_weight = np.concatenate(([0.5], np.ones(nz-1), [0.5])) * (-1) ** np.arange(nz+1)
    
    for j in range(len(z_int)): 
        diff = z_int[j] - z
        temp = bary_weight / diff
        Cint[j, :] = temp / np.sum(temp)
        
        if np.min(np.abs(diff)) == 0:
            Cint[j, :] = 0
            Cint[j, np.where(diff == 0)[0]] = 1
    
    Cint = Cint[:, 1:-1]
    
    C_int = np.block([
        [Cint, np.zeros((len(z_int), nz-1)), np.zeros((len(z_int), nz-1))],
        [np.zeros((len(z_int), nz-1)), Cint, np.zeros((len(z_int), nz-1))],
        [np.zeros((len(z_int), nz-1)), np.zeros((len(z_int), nz-1)), Cint]
    ])
    
    C = np.dot(C_int, C_ori)
    
    return A, B, C