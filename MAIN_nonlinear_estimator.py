#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 11:17:04 2024

@author: jitongd
"""

import numpy as np
from cheb_numeric import cheb
from derivative_calculation import fft_xy, ifft_xy, get_3d_physical
from create_state_space_model import create_ss_model_d1, create_ss_model_noKF
from read_file import get_uvw
from scipy.signal import lsim
from estimation_functions import get_numpy_array_from_results, get_nonlinear_forcing
import ray

@ray.remote
def get_new_velocity_ssrom(kx,ky,dkx,dky,nx,vel_3F,nonlinear_forcing_3F,nz,Retau, channelRe, z_whole, whe_eddyv, Ts):
    if kx == 0 and ky == 0:
        return

    kx_index = int(kx/dkx * (kx>=0) + (nx - kx/dkx) * (kx<0))
    ky_index = int(ky/dky)
    
    vel_3F_one_wavenumber = np.squeeze(vel_3F[:,ky_index,kx_index])
    nonlinear_forcing_3F_one_wavenumber = np.squeeze(nonlinear_forcing_3F[:,ky_index,kx_index])
    
    T = np.block([
        [np.zeros((nz-1, nz-1)), np.zeros((nz-1, nz-1)), np.eye(nz-1)],
        [1j * ky * np.eye(nz-1), -1j * kx * np.eye(nz-1), np.zeros((nz-1, nz-1))]
        ])
    
    x_old = T @ vel_3F_one_wavenumber
    
    _, chan_whole_d = create_ss_model_noKF(kx, ky, nz, Retau, channelRe, z_whole, whe_eddyv, Ts)
    
    x_new = chan_whole_d.A @ x_old + chan_whole_d.B @ nonlinear_forcing_3F_one_wavenumber
    
    vel_3F_one_wavenumber = chan_whole_d.C @ x_new
    
    return vel_3F_one_wavenumber, kx, ky


@ray.remote
def get_estimation(kx,ky,dkx,dky,nx,nz,z_mea_index,measurement,vel_3F,nonlinear_forcing_3F, Retau, channelRe, z_mea, z_whole, whe_eddyv, dist_var, noise_var,Ts):
    if kx == 0 and ky == 0:
        return

    kx_index = int(kx/dkx * (kx>=0) + (nx - kx/dkx) * (kx<0))
    ky_index = int(ky/dky)
    
    measurement = np.squeeze(measurement[:,ky_index,kx_index])
    
    vel_3F_one_wavenumber = np.squeeze(vel_3F[:,ky_index,kx_index])
    nonlinear_forcing_3F_one_wavenumber = np.squeeze(nonlinear_forcing_3F[:,ky_index,kx_index])
    
    T = np.block([
        [np.zeros((nz-1, nz-1)), np.zeros((nz-1, nz-1)), np.eye(nz-1)],
        [1j * ky * np.eye(nz-1), -1j * kx * np.eye(nz-1), np.zeros((nz-1, nz-1))]
        ])
    
    x_old = T @ vel_3F_one_wavenumber
    
    chan_whole_d, L_d, C_mea = create_ss_model_d1(kx, ky, nz, Retau, channelRe, z_mea, z_whole, whe_eddyv, dist_var, noise_var,Ts)
    
    # state update
    x_1 = chan_whole_d.A @ x_old + chan_whole_d.B @ nonlinear_forcing_3F_one_wavenumber
    
    # measurement update
    vel_3F_one_wavenumber = x_1 + L_d @ (measurement - C_mea @ x_1)
    
    return vel_3F_one_wavenumber, kx, ky



if __name__ == '__main__':
    workers = 2
    ray.init(num_cpus=workers)
    assert ray.is_initialized()
    
    Retau = 180
    
    filename = f'full{Retau}_mean.npz'
    data = np.load(filename, allow_pickle=True)
    channelRe = data['channelRe'].item()
    nx = data['nx'].item()
    ny = data['ny'].item()
    nz = data['nz'].item()
    nzDNS = data['nzDNS'].item()
    dkx = data['dkx'].item()
    dky = data['dky'].item()
    nx_d = data['nx_d'].item()
    ny_d = data['ny_d'].item()
    xu = data['xu']
    xp = data['xp']
    yv = data['yv']
    yp = data['yp']
    zw = data['zw']
    zp = data['zp']
    
    
    if Retau == 180:
        loadname1 = '180/112x112x150'
        nx_d = 50
        ny_d = 50
    elif Retau == 395:
        loadname1 = '395/256x256x300'
    elif Retau == 590:
        loadname1 = '590/384x384x500'
        nx_d = 180
        ny_d = 180
    else:
        raise ValueError("Unsupported Retau value")
        
    
    _,zc = cheb(nz)
    Diff,z_whole = cheb(nz)
    z_whole   = z_whole[1:-1]
    Diff = Diff[1:-1,1:-1]
    z_mea_index = np.array([16,50,111])
    z_mea     = z_whole[z_mea_index]
    whe_eddyv = 0
    dist_var = 1
    noise_var = 1e-6
    Ts = 0.002
    Tm = 0.02
    
    N_wave = 2
    kx = np.concatenate((np.arange(0, N_wave*dkx+1, dkx), np.arange(-N_wave*dkx, 0, dkx)))
    ky = np.arange(0, N_wave*dky+1, dky)
    kx_array = np.repeat(kx, len(ky))
    ky_array = np.tile(ky, len(kx))
    
    read_start    = 60000
    read_end      = 70000
    read_interval = 10000
    read_array    = np.arange(read_start,read_end+1,read_interval)
    t_array = np.arange(0,Ts*len(read_array),Tm)
    
    
    # obtain the velocity data for measurements
    true_uF = np.zeros((nz-1,ny,nx,len(read_array)))
    true_vF = np.zeros((nz-1,ny,nx,len(read_array)))
    true_wF = np.zeros((nz-1,ny,nx,len(read_array)))
    
    for k_index in range(len(read_array)):
        loadname_u = 'u/u_it{}.dat'.format(read_array[k_index])
        loadname_v = 'v/v_it{}.dat'.format(read_array[k_index])
        loadname_w = 'w/w_it{}.dat'.format(read_array[k_index])
        
        u,v,w = get_uvw(xu,xp,yv,yp,zp,zc,zw,nzDNS,ny,nx,loadname_u,loadname_v,loadname_w)
        
        u_F = ifft_xy(u)
        v_F = ifft_xy(v)
        w_F = ifft_xy(w)
        
        true_uF[:,:,:,k_index] = u_F
        true_vF[:,:,:,k_index] = v_F
        true_wF[:,:,:,k_index] = w_F
    
    
    vel_3F = np.zeros((3*(nz-1),ny,nx))
    nonlinear_forcing_3F = vel_3F = np.zeros((3*(nz-1),ny,nx))
    vel_3F_array = np.zeros((3*(nz-1),ny,nx),len(read_array))
    
    # begin estimation
    for k_time in range(len(read_array)):
        
        print(k_time)
        
        # SSROM simulation
        for k_time1 in range(int(Tm/Ts)):
            
            result_ids = [get_new_velocity_ssrom.remote(kx,ky,dkx,dky,N_wave,vel_3F,nonlinear_forcing_3F,nz,Retau, channelRe, z_whole, whe_eddyv, Ts)
                          for kx in range(0,dkx*N_wave+1,dkx) for ky in range(0,dky*N_wave+1,dky)]
            
            results = ray.get(result_ids) 
            
            vel_3F = get_numpy_array_from_results(results,nz,ny,nx,dkx,dky)
            
            if np.isnan(vel_3F).any():
                print(f"yeah, it diverges again. {(k_time-1)*(Tm/Ts) + k_time1}")
                break
            
            vel_3 = ifft_xy(vel_3F)
            
            u = vel_3[:(nz-1),:,:]
            v = vel_3[(nz-1):2*(nz-1),:,:]
            w = vel_3[2*(nz-1):3*(nz-1),:,:]
            
            f_x,f_y,f_z = get_nonlinear_forcing(u, v, w, Diff, nx_d, ny_d, nx, ny, dkx, dky)
            
            nonlinear_forcing_3 = np.stack((f_x, f_y, f_z), axis=0)
            
            nonlinear_forcing_3F = fft_xy(nonlinear_forcing_3)
            
            
        # Kalman filter estimation
        
        true_uF_one_mea = np.squeeze(true_uF[z_mea_index,:,:,k_time])
        true_vF_one_mea = np.squeeze(true_vF[z_mea_index,:,:,k_time])
        true_wF_one_mea = np.squeeze(true_wF[z_mea_index,:,:,k_time])

        measurement = np.stack((true_uF_one_mea,true_vF_one_mea,true_wF_one_mea),axis=0)
        
        results_id = [get_estimation(kx,ky,dkx,dky,nx,nz,z_mea_index,measurement,vel_3F,nonlinear_forcing_3F, Retau, channelRe, z_mea, z_whole, whe_eddyv, dist_var, noise_var,Ts)
                      for kx in range(0,dkx*N_wave+1,dkx) for ky in range(0,dky*N_wave+1,dky)]
        
        results = ray.get(result_ids) 
        
        vel_3F = get_numpy_array_from_results(results,nz,ny,nx,dkx,dky)
        
        vel_3F_array[:,:,:,k_time] = vel_3F
        
        vel_3 = ifft_xy(vel_3F)
        
        u = vel_3[:(nz-1),:,:]
        v = vel_3[(nz-1):2*(nz-1),:,:]
        w = vel_3[2*(nz-1):3*(nz-1),:,:]
        
        f_x,f_y,f_z = get_nonlinear_forcing(u, v, w, Diff, nx_d, ny_d, nx, ny, dkx, dky)
        
        nonlinear_forcing_3 = np.stack((f_x, f_y, f_z), axis=0)
        
        nonlinear_forcing_3F = fft_xy(nonlinear_forcing_3)
        
        
    ray.shutdown()
    assert not ray.is_initialized()

        
        
        
    
    
    

