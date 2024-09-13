#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:41:59 2024

@author: jitongd
"""

import numpy as np
from cheb_numeric import cheb
from derivative_calculation import fft_xy, ifft_xy
from create_state_space_model import create_ss_model_lin
from read_file import get_uvw
from scipy.signal import lsim
import ray

@ray.remote
def get_estimation_one_wavenumber(kx,ky,N_wave,dkx,dky,z_mea_index,nz, Retau, channelRe, z_whole, whe_eddyv, dist_var, noise_var,Ts,t_array,true_uF,true_vF,true_wF):
    
    if kx == 0 and ky == 0:
        return

    kx_index = int(kx/dkx * (kx>=0) + (2*N_wave + 2 - kx/dkx) * (kx<0))
    ky_index = int(ky/dky)

    true_uF_one_mea = np.squeeze(true_uF[z_mea_index,ky_index,kx_index,:]).conj().T
    true_vF_one_mea = np.squeeze(true_vF[z_mea_index,ky_index,kx_index,:]).conj().T
    true_wF_one_mea = np.squeeze(true_wF[z_mea_index,ky_index,kx_index,:]).conj().T

    z_mea = z_whole[z_mea_index]

    chan_whole_c, _ = create_ss_model_lin(kx, ky, nz, Retau, channelRe, z_mea, z_whole, whe_eddyv, dist_var, noise_var,Ts)

    input = np.hstack([np.zeros((len(t_array),nz-1)), 
                      np.zeros((len(t_array),nz-1)),
                      np.zeros((len(t_array),nz-1)),
                      true_uF_one_mea,
                      true_vF_one_mea,
                      true_wF_one_mea
                      ])

    _, vel_est, _ = lsim(chan_whole_c,input,t_array)
    
    return vel_est
    
    

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
    _,z_whole = cheb(nz)
    z_whole   = z_whole[1:-1]
    z_mea_index = np.array([16,50,111])
    whe_eddyv = 0
    dist_var = 1
    noise_var = 1e-6
    Ts = 0.002
    
    N_wave = 2
    kx = np.concatenate((np.arange(0, N_wave*dkx+1, dkx), np.arange(-N_wave*dkx, 0, dkx)))
    ky = np.arange(0, N_wave*dky+1, dky)
    kx_array = np.repeat(kx, len(ky))
    ky_array = np.tile(ky, len(kx))
    
    read_start    = 60000
    read_end      = 70000
    read_interval = 10000
    read_array    = np.arange(read_start,read_end+1,read_interval)
    t_array = np.arange(0,Ts*len(read_array),Ts)
    
    true_uF = np.zeros((nz-1,N_wave+1,2*N_wave+1,len(read_array)))
    true_vF = np.zeros((nz-1,N_wave+1,2*N_wave+1,len(read_array)))
    true_wF = np.zeros((nz-1,N_wave+1,2*N_wave+1,len(read_array)))
    
    for k_index in range(len(read_array)):
        loadname_u = 'u/u_it{}.dat'.format(read_array[k_index])
        loadname_v = 'v/v_it{}.dat'.format(read_array[k_index])
        loadname_w = 'w/w_it{}.dat'.format(read_array[k_index])
        
        u,v,w = get_uvw(xu,xp,yv,yp,zp,zc,zw,nzDNS,ny,nx,loadname_u,loadname_v,loadname_w)
        
        u_F = ifft_xy(u)
        v_F = ifft_xy(v)
        w_F = ifft_xy(w)
        
        true_uF[:,:,:,k_index] = u[:, :N_wave+1, np.concatenate((np.arange(0, N_wave+1,1), np.arange(nx-N_wave,nx,1)))]
        true_vF[:,:,:,k_index] = v[:, :N_wave+1, np.concatenate((np.arange(0, N_wave+1,1), np.arange(nx-N_wave,nx,1)))]
        true_wF[:,:,:,k_index] = w[:, :N_wave+1, np.concatenate((np.arange(0, N_wave+1,1), np.arange(nx-N_wave,nx,1)))]
    
    
    result_ids = [get_estimation_one_wavenumber.remote(kx,ky,N_wave,dkx,dky,z_mea_index,nz, Retau, channelRe, z_whole, whe_eddyv, dist_var, noise_var,Ts,t_array,true_uF,true_vF,true_wF)
                  for kx in range(0,dkx*N_wave+1,dkx) for ky in range(0,dky*N_wave+1,dky)]
    
    results = ray.get(result_ids) 
    
    ray.shutdown()
    assert not ray.is_initialized()





#%%
import numpy as np
from cheb_numeric import cheb
from derivative_calculation import fft_xy, ifft_xy
from create_state_space_model import create_ss_model_lin
from read_file import get_uvw
from scipy.signal import lsim

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
_,z_whole = cheb(nz)
z_whole   = z_whole[1:-1]
z_mea_index = np.array([16,50,111])
whe_eddyv = 0
dist_var = 1
noise_var = 1e-6
Ts = 0.002

N_wave = 2
kx = np.concatenate((np.arange(0, N_wave*dkx+1, dkx), np.arange(-N_wave*dkx, 0, dkx)))
ky = np.arange(0, N_wave*dky+1, dky)
kx_array = np.repeat(kx, len(ky))
ky_array = np.tile(ky, len(kx))

read_start    = 60000
read_end      = 70000
read_interval = 10000
read_array    = np.arange(read_start,read_end+1,read_interval)
t_array = np.arange(0,Ts*len(read_array),Ts)

true_uF = np.zeros((nz-1,N_wave+1,2*N_wave+1,len(read_array)))
true_vF = np.zeros((nz-1,N_wave+1,2*N_wave+1,len(read_array)))
true_wF = np.zeros((nz-1,N_wave+1,2*N_wave+1,len(read_array)))

for k_index in range(len(read_array)):
    loadname_u = 'u/u_it{}.dat'.format(read_array[k_index])
    loadname_v = 'v/v_it{}.dat'.format(read_array[k_index])
    loadname_w = 'w/w_it{}.dat'.format(read_array[k_index])
    
    u,v,w = get_uvw(xu,xp,yv,yp,zp,zc,zw,nzDNS,ny,nx,loadname_u,loadname_v,loadname_w)
    
    u_F = ifft_xy(u)
    v_F = ifft_xy(v)
    w_F = ifft_xy(w)
    
    true_uF[:,:,:,k_index] = u[:, :N_wave+1, np.concatenate((np.arange(0, N_wave+1,1), np.arange(nx-N_wave,nx,1)))]
    true_vF[:,:,:,k_index] = v[:, :N_wave+1, np.concatenate((np.arange(0, N_wave+1,1), np.arange(nx-N_wave,nx,1)))]
    true_wF[:,:,:,k_index] = w[:, :N_wave+1, np.concatenate((np.arange(0, N_wave+1,1), np.arange(nx-N_wave,nx,1)))]

kx = 0
ky = 2

kx_index = int(kx/dkx * (kx>=0) + (2*N_wave + 2 - kx/dkx) * (kx<0))
ky_index = int(ky/dky)

true_uF_one_mea = np.squeeze(true_uF[z_mea_index,ky_index,kx_index,:]).conj().T
true_vF_one_mea = np.squeeze(true_vF[z_mea_index,ky_index,kx_index,:]).conj().T
true_wF_one_mea = np.squeeze(true_wF[z_mea_index,ky_index,kx_index,:]).conj().T

z_mea = z_whole[z_mea_index]

chan_whole_c, _ = create_ss_model_lin(kx, ky, nz, Retau, channelRe, z_mea, z_whole, whe_eddyv, dist_var, noise_var,Ts)

input = np.hstack([np.zeros((len(t_array),nz-1)), 
                  np.zeros((len(t_array),nz-1)),
                  np.zeros((len(t_array),nz-1)),
                  true_uF_one_mea,
                  true_vF_one_mea,
                  true_wF_one_mea
                  ])

_, vel_est, _ = lsim(chan_whole_c,input,t_array)


