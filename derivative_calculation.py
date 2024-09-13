#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 19:51:17 2024

@author: jitongd
"""

import numpy as np

# calculate the 2D Fourier transform in the xy plane, 'a' has the shape of [z,y,x]
def fft_xy(a):
    aF = np.fft.fft2(a) / a.shape[2] / a.shape[1]
    return aF

# calculate the 2D inverse Fourier transform in the xy plane, 'aF' has the shape of [z,y,x]
def ifft_xy(aF):
    a = np.fft.ifft2(aF) * aF.shape[2] * aF.shape[1]
    return a

# get the streamwise and spanwise wavenumber vectors, nx_d and ny_d specify the number of streamwise and spanwise wavenumbers
def get_kx_array_ky_array(nx_d,ny_d,dkx,dky):
    kx = np.concatenate((np.arange(0, nx_d*dkx+1, dkx), np.arange(-nx_d*dkx, 0, dkx)))
    ky = np.arange(0, ny_d*dky+1, dky)
    kx_array = np.repeat(kx[:], len(ky), axis=0)
    ky_array = np.tile(ky, len(kx))
    return kx_array, ky_array

# get the streamwise and spanwise wavenumber matrices used for calculating the derivatives in the streamwise and spanwise directions
def getkm(alpha_array, beta_array, nx, ny, dkx, dky):
    kx_m = np.zeros((ny, nx))
    ky_m = np.zeros((ny, nx))
    
    for k_array in range(len(alpha_array)):
        index_kx = round(alpha_array[k_array] / dkx) 
        index_ky = round(beta_array[k_array] / dky) 
        
        if alpha_array[k_array] < 0:
            kx_m[index_ky, nx + index_kx] = alpha_array[k_array]
            ky_m[index_ky, nx + index_kx] = beta_array[k_array]
        else:
            kx_m[index_ky, index_kx] = alpha_array[k_array]
            ky_m[index_ky, index_kx] = beta_array[k_array]
    
    kx_m[ny-1:ny//2+1:-1, :] = kx_m[1:ny//2-1, :]
    ky_m[ny-1:ny//2+1:-1, :] = -ky_m[1:ny//2-1, :]
    
    return kx_m, ky_m

# get the Fourier coefficients of the derivatives in three directions
def get_3dF(f, Diff, kx_m, ky_m):
    f = np.transpose(f, (1, 2, 0))
    dfdx = 1j * kx_m[:, :, np.newaxis] * f
    dfdy = 1j * ky_m[:, :, np.newaxis] * f
    
    f = np.transpose(f, (2, 0, 1))
    dfdx = np.transpose(dfdx, (2, 0, 1))
    dfdy = np.transpose(dfdy, (2, 0, 1))
    
    dfdz = np.tensordot(Diff, f, axes=(1, 0))
    
    return dfdx, dfdy, dfdz


# get the derivatives in three directions in physical space
def get_3d_physical(u, Diff, nx_d, ny_d, nx, ny, dkx, dky):
    alpha_array, beta_array = get_kx_array_ky_array(nx_d,ny_d,dkx,dky)
    kx_m, ky_m = getkm(alpha_array, beta_array, nx, ny, dkx, dky)
    uF = fft_xy(u)
    duFdx, duFdy, duFdz = get_3dF(uF,Diff,kx_m,ky_m)
    dudx = ifft_xy(duFdx).real
    dudy = ifft_xy(duFdy).real
    dudz = ifft_xy(duFdz).real
    
    return dudx, dudy, dudz


def get_velocity_tensor(u, v, w, Diff, nx_d, ny_d, nx, ny, dkx, dky):
    dudx, dudy, dudz = get_3d_physical(u, Diff, nx_d, ny_d, nx, ny, dkx, dky)
    dvdx, dvdy, dvdz = get_3d_physical(v, Diff, nx_d, ny_d, nx, ny, dkx, dky)
    dwdx, dwdy, dwdz = get_3d_physical(w, Diff, nx_d, ny_d, nx, ny, dkx, dky)
    velocity_tensor = np.stack((dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz), axis=3)

    return velocity_tensor




    




