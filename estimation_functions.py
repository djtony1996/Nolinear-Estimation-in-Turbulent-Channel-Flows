#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:58:02 2024

@author: jitongd
"""
import numpy as np
from derivative_calculation import get_3d_physical


def get_numpy_array_from_results(results,nz,ny,nx,dkx,dky):
    
    numpy_array = np.zeros((nz-1,ny,nx))
    
    for k_index in range(len(results)):
        try:
            kx = results[k_index][1]
            ky = results[k_index][2]
            
            kx_index = int(kx/dkx * (kx>=0) + (nx - kx/dkx) * (kx<0))
            ky_index = int(ky/dky)
            
            numpy_array[:,ky_index,kx_index] = results[k_index][0]
        except:
            print('(0,0)')
            
    numpy_array[:,0,(nx//2-2)::-1] = np.conj(numpy_array[0, 1:(nx//2)])
    numpy_array[:,(ny//2-2)::-1,0] = np.conj(numpy_array[1:(ny//2), 0])
    numpy_array[:,(ny//2-2)::-1, (nx//2-2)::-1] = np.conj(numpy_array[1:(ny//2), 1:(nx//2)])
    numpy_array[:,(ny//2-2)::-1, 1:(nx//2)]     = np.conj(numpy_array[1:(ny//2), (nx//2-2)::-1])
    
    return numpy_array


def get_nonlinear_forcing(u, v, w, Diff, nx_d, ny_d, nx, ny, dkx, dky):
    dudx, dudy, dudz = get_3d_physical(u, Diff, nx_d, ny_d, nx, ny, dkx, dky)
    dvdx, dvdy, dvdz = get_3d_physical(v, Diff, nx_d, ny_d, nx, ny, dkx, dky)
    dwdx, dwdy, dwdz = get_3d_physical(w, Diff, nx_d, ny_d, nx, ny, dkx, dky)
    
    f_x = -u*dudx - v*dudy - w*dudz
    f_y = -u*dvdx - v*dvdy - w*dvdz
    f_z = -u*dwdx - v*dwdy - w*dwdz
    
    return f_x, f_y, f_z


