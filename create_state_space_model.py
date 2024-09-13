#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 21:14:48 2024

@author: jitongd
"""

import numpy as np
from OSS_operator import OSS_operator
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.signal import StateSpace, cont2discrete

def create_ss_model_noKF(kx, ky, nz, Retau, channelRe, z_mea, z_whole, whe_eddyv, Ts):
    
    A,B,C_whole = OSS_operator(kx, ky, nz, Retau, channelRe, z_whole, whe_eddyv)
    
    chan_whole_c = StateSpace(A, B, C_whole, np.zeros((C_whole.shape[0],B.shape[1])))
    temp_chan_whole_d = cont2discrete((A, B, C_whole, np.zeros((C_whole.shape[0],B.shape[1]))), Ts)
    A_d, B_d, C_d, D_d, _ = temp_chan_whole_d
    chan_whole_d = StateSpace(A_d, B_d, C_d, D_d, dt=Ts)
    del A_d, B_d, C_d, D_d, temp_chan_whole_d
    
    return chan_whole_c, chan_whole_d


def create_ss_model_lin(kx, ky, nz, Retau, channelRe, z_mea, z_whole, whe_eddyv, dist_var, noise_var,Ts):
    
    A,B,C_mea   = OSS_operator(kx, ky, nz, Retau, channelRe, z_mea, whe_eddyv)
    
    A,B,C_whole = OSS_operator(kx, ky, nz, Retau, channelRe, z_whole, whe_eddyv)
    
    Q = dist_var * np.eye(3*(nz-1))
    R = noise_var * np.eye(3*len(z_mea))
    
    P = solve_continuous_are(A.conj().T, C_mea.conj().T, B @ Q @ B.conj().T, R) 
    L = P @ C_mea.conj().T @ np.linalg.inv(R)
    
    new_B = np.hstack([B,L])
    chan_whole_c = StateSpace(A-L@C_mea, new_B, C_whole, np.zeros((C_whole.shape[0],new_B.shape[1])))
    temp_chan_whole_d = cont2discrete((A-L@C_mea, new_B, C_whole, np.zeros((C_whole.shape[0],new_B.shape[1]))), Ts)
    A_d, B_d, C_d, D_d, _ = temp_chan_whole_d
    chan_whole_d = StateSpace(A_d, B_d, C_d, D_d, dt=Ts)
    del A_d, B_d, C_d, D_d, temp_chan_whole_d
    
    return chan_whole_c, chan_whole_d



def create_ss_model_d1(kx, ky, nz, Retau, channelRe, z_mea, z_whole, whe_eddyv, dist_var, noise_var,Ts):
    
    A,B,C_mea   = OSS_operator(kx, ky, nz, Retau, channelRe, z_mea, whe_eddyv)
    
    A,B,C_whole = OSS_operator(kx, ky, nz, Retau, channelRe, z_whole, whe_eddyv)
    
    Q = dist_var * np.eye(3*(nz-1))
    R = noise_var * np.eye(3*len(z_mea))
    
    temp_chan_whole_d = cont2discrete((A, B, C_whole, np.zeros((C_whole.shape[0],B.shape[1]))), Ts)
    A_d, B_d, C_d, D_d, _ = temp_chan_whole_d

    P_d = solve_discrete_are(A_d.conj().T, C_mea.conj().T, B_d @ Q @ B_d.conj().T, R) 
    L_d = A_d @ P_d @ C_mea.conj().T @ np.linalg.inv(C_mea @ P_d @ C_mea.conj().T + R)

    new_B_d = np.hstack([B_d,L_d])
    chan_whole_d = StateSpace(A_d-L_d@C_mea, new_B_d, C_d, np.zeros((C_d.shape[0],new_B_d.shape[1])))
    
    return chan_whole_d

