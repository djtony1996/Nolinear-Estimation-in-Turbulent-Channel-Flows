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

def create_ss_model_noKF(kx, ky, nz, Retau, channelRe, z_whole, whe_eddyv, Ts):
    
    A,B,C_whole = OSS_operator(kx, ky, nz, Retau, channelRe, z_whole, whe_eddyv)
    
    """
    a state-space model is of this form:
        dx/dt = Ax + Bu
        y     = Cx + Du
    x -- state vector
    y -- output vector
    u -- input vector
    """
    
    chan_whole_c = StateSpace(A, B, C_whole, np.zeros((C_whole.shape[0],B.shape[1])))
    temp_chan_whole_d = cont2discrete((A, B, C_whole, np.zeros((C_whole.shape[0],B.shape[1]))), Ts)
    A_d, B_d, C_d, D_d, _ = temp_chan_whole_d
    chan_whole_d = StateSpace(A_d, B_d, C_d, D_d, dt=Ts)
    del A_d, B_d, C_d, D_d, temp_chan_whole_d
    
    return chan_whole_c, chan_whole_d


def create_ss_model_lin(kx, ky, nz, Retau, channelRe, z_mea, z_whole, whe_eddyv, dist_var, noise_var,Ts):
    
    A,B,C_mea   = OSS_operator(kx, ky, nz, Retau, channelRe, z_mea, whe_eddyv)
    
    A,B,C_whole = OSS_operator(kx, ky, nz, Retau, channelRe, z_whole, whe_eddyv)
    
    """
    the continuous-time state-space model considering the disturbance w and measurement noise v is:
        dx/dt = Ax + Bu + Bw
        y     = Cx + Du + v
    w -- disturbance
    v -- measurement noise
    
    Q is the disturbance covariance matrix, Q = E(ww^H)
    R is the measurement noise covariance matrix, R = E(vv^H)
    """
    
    Q = dist_var * np.eye(3*(nz-1))
    R = noise_var * np.eye(3*len(z_mea))
    
    """
    In Matlab, you can use command 'lqe', type 'help lqe' in the console for more explanation (no website for this command)
    
    P -- steady-state error covariance
    P is obtained by solving the continuous-time algebraic Riccati equation (CARE)
    in Matlab, the equation is solved using command 'icare'
        XA + A^H X - XBR^{-1} B^H X + Q = 0
    in this case, need to use A^H for A and BQB^H for Q
    
    L -- Kalman filter gain
    L is obtained by:
        L = P C^H R^{-1} 
    
    """
    
    P = solve_continuous_are(A.conj().T, C_mea.conj().T, B @ Q @ B.conj().T, R) 
    L = P @ C_mea.conj().T @ np.linalg.inv(R)
    
    """
    The state-space model the estimated state vector \hat{x} is:
        d\hat{x}/dt = A\hat{x} + Bu + L(y-C\hat{x}-Du)
    """
    
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
    
    """
    the discrete-time state-space model considering the disturbance w and measurement noise v is:
        x[n+1] = Ax[n] + Bu[n] + Bw[n]
        y[n]   = Cx[n] + Du[n] + v[n]
    w -- disturbance
    v -- measurement noise
    
    Q is the disturbance covariance matrix, Q = E(w[n]w[n]^H)
    R is the measurement noise covariance matrix, R = E(v[n]v[n]^H)
    """
    
    Q = dist_var * np.eye(3*(nz-1))
    R = noise_var * np.eye(3*len(z_mea))
    
    """
    to transfer the continuous-time state-space models into a discrete-time state-space model
    in Matlab, you can use command 'c2d'
    """
    
    temp_chan_whole_d = cont2discrete((A, B, C_whole, np.zeros((C_whole.shape[0],B.shape[1]))), Ts)
    A_d, B_d, C_d, D_d, _ = temp_chan_whole_d
    
    """
    In Matlab, you can use command 'dlqe', type 'help dlqe' in the console for more explanation (no website for this command)
    
    P -- steady-state error covariance
    P is obtained by solving the discrete-time algebraic Riccati equation (DARE)
    in Matlab, the equation is solved using command 'idare'
        A^H XA - X - (A^H XB)(R + B^H XB)^{-1} (B^H XA) + Q = 0
    in this case, need to use A^H for A and BQB^H for Q
    
    L -- Kalman filter gain
    L is obtained by:
        L = APC^H (CPC^H + R)^{-1}
    
    """
    
    P_d = solve_discrete_are(A_d.conj().T, C_mea.conj().T, B_d @ Q @ B_d.conj().T, R) 
    L_d = A_d @ P_d @ C_mea.conj().T @ np.linalg.inv(C_mea @ P_d @ C_mea.conj().T + R)

    new_B_d = np.hstack([B_d,L_d])
    chan_whole_d = StateSpace(A_d-L_d@C_mea, new_B_d, C_d, np.zeros((C_d.shape[0],new_B_d.shape[1])))
    
    return chan_whole_d, L_d, C_mea

