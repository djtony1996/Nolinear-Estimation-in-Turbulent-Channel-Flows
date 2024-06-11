#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the basic data and store them into a file for each Reynolds number. 
The basic data include the DNS grid, mean velocity, Cess eddy viscosity and wavenumbers. 

@author: jitongd
"""

#%% Re_{\tau}=180
import numpy as np
from cheb_numeric import *
from read_file import *

M_means = np.loadtxt('chan180.means.txt')
z_mkm = np.concatenate((1 - M_means[:-1, 0], np.flipud(M_means[:, 0] - 1)))
U_mkm = np.concatenate((M_means[:-1, 2], np.flipud(M_means[:, 2])))

Diff, zc = cheb(128)
U_mkm_diff1 = np.dot(Diff, U_mkm)
U_mkm_diff2 = np.dot(Diff, U_mkm_diff1)

channelRe = {}
channelRe['Up'] = U_mkm
channelRe['Up_diff1'] = U_mkm_diff1
channelRe['Up_diff2'] = U_mkm_diff2
channelRe['Ret'] = 180

kappa = 0.426
A_const = 25.4
temp_z = 1 - zc[:65]
temp = np.sqrt(1 + kappa**2 * (channelRe['Ret'])**2 / 9 * (2 * temp_z - temp_z**2)**2 * (3 - 4 * temp_z + 2 * temp_z**2)**2 * (1 - np.exp(-(temp_z) * (channelRe['Ret']) / A_const))**2) / 2 + 1 / 2
channelRe['vTpd'] = np.concatenate((temp, np.flipud(temp[:-1])))
channelRe['vTp_diff1'] = np.dot(Diff, channelRe['vTpd'])
channelRe['vTp_diff2'] = np.dot(Diff, channelRe['vTp_diff1'])

nx = 112
dkx = 1
Lx = 2 * np.pi
ny = 112
dky = 2
Ly = np.pi
nz = 128
nzDNS = 150
xgrid = np.linspace(0, Lx, nx + 1)
ygrid = np.linspace(0, Ly, ny + 1)
xu = xgrid[1:]
xp = (xgrid[:-1] + xu) / 2
yv = ygrid[1:]
yp = (ygrid[:-1] + yv) / 2

dz = np.squeeze(read_bin('inputdir/dz_grid_180.dat', (nzDNS, 1)))
zp = get_zp(dz)
zp = np.concatenate(([-1], zp - 1, [1]))
_, zw = cheb(nzDNS)
zw = np.flipud(zw)

kx = np.concatenate((np.arange(11), np.arange(-10, 0)))
ky = np.arange(0, 21, 2)
kx_array = np.repeat(kx[:], len(ky), axis=0)
ky_array = np.tile(ky, len(kx))

np.savez('full180_mean.npz', channelRe=channelRe, nx=nx, ny=ny, nz=nz, nzDNS=nzDNS, dkx=dkx, dky=dky, Lx=Lx, Ly=Ly, xu=xu, xp=xp, yv=yv, yp=yp, zw=zw, zp=zp, kx_array=kx_array, ky_array=ky_array)


#%% Re_{\tau}=395

import numpy as np
from cheb_numeric import *
from read_file import *

M_means = np.loadtxt('chan395.means.txt')
z_mkm = np.concatenate((1 - M_means[:-1, 0], np.flipud(M_means[:, 0] - 1)))
U_mkm = np.concatenate((M_means[:-1, 2], np.flipud(M_means[:, 2])))

Diff, zc = cheb(256)
U_mkm_diff1 = np.dot(Diff, U_mkm)
U_mkm_diff2 = np.dot(Diff, U_mkm_diff1)

channelRe = {}
channelRe['Up'] = U_mkm
channelRe['Up_diff1'] = U_mkm_diff1
channelRe['Up_diff2'] = U_mkm_diff2
channelRe['Ret'] = 395

kappa = 0.426
A_const = 25.4
temp_z = 1 - zc[:129]
temp = np.sqrt(1 + kappa**2 * (channelRe['Ret'])**2 / 9 * (2 * temp_z - temp_z**2)**2 * (3 - 4 * temp_z + 2 * temp_z**2)**2 * (1 - np.exp(-(temp_z) * (channelRe['Ret']) / A_const))**2) / 2 + 1 / 2
channelRe['vTpd'] = np.concatenate((temp, np.flipud(temp[:-1])))
channelRe['vTp_diff1'] = np.dot(Diff, channelRe['vTpd'])
channelRe['vTp_diff2'] = np.dot(Diff, channelRe['vTp_diff1'])

nx = 256
dkx = 1
Lx = 2 * np.pi
ny = 256
dky = 2
Ly = np.pi
nz = 256
nzDNS = 300
xgrid = np.linspace(0, Lx, nx + 1)
ygrid = np.linspace(0, Ly, ny + 1)
xu = xgrid[1:]
xp = (xgrid[:-1] + xu) / 2
yv = ygrid[1:]
yp = (ygrid[:-1] + yv) / 2

dz = np.squeeze(read_bin('inputdir/dz_grid_395.dat', (nzDNS, 1)))
zp = get_zp(dz)
zp = np.concatenate(([-1], zp - 1, [1]))
_, zw = cheb(nzDNS)
zw = np.flipud(zw)

kx = np.concatenate((np.arange(11), np.arange(-10, 0)))
ky = np.arange(0, 21, 2)
kx_array = np.repeat(kx[:], len(ky), axis=0)
ky_array = np.tile(ky, len(kx))

np.savez('full395_mean.npz', channelRe=channelRe, nx=nx, ny=ny, nz=nz, nzDNS=nzDNS, dkx=dkx, dky=dky, Lx=Lx, Ly=Ly, xu=xu, xp=xp, yv=yv, yp=yp, zw=zw, zp=zp, kx_array=kx_array, ky_array=ky_array)




#%% Re_{\tau}=590

import numpy as np
from cheb_numeric import *
from read_file import *

M_means = np.loadtxt('chan590.means.txt')
z_mkm = np.concatenate((1 - M_means[:-1, 0], np.flipud(M_means[:, 0] - 1)))
U_mkm = np.concatenate((M_means[:-1, 2], np.flipud(M_means[:, 2])))

Diff, zc = cheb(256)
U_mkm_diff1 = np.dot(Diff, U_mkm)
U_mkm_diff2 = np.dot(Diff, U_mkm_diff1)

channelRe = {}
channelRe['Up'] = U_mkm
channelRe['Up_diff1'] = U_mkm_diff1
channelRe['Up_diff2'] = U_mkm_diff2
channelRe['Ret'] = 590

kappa = 0.426
A_const = 25.4
temp_z = 1 - zc[:129]
temp = np.sqrt(1 + kappa**2 * (channelRe['Ret'])**2 / 9 * (2 * temp_z - temp_z**2)**2 * (3 - 4 * temp_z + 2 * temp_z**2)**2 * (1 - np.exp(-(temp_z) * (channelRe['Ret']) / A_const))**2) / 2 + 1 / 2
channelRe['vTpd'] = np.concatenate((temp, np.flipud(temp[:-1])))
channelRe['vTp_diff1'] = np.dot(Diff, channelRe['vTpd'])
channelRe['vTp_diff2'] = np.dot(Diff, channelRe['vTp_diff1'])

nx = 384
dkx = 1
Lx = 2 * np.pi
ny = 384
dky = 2
Ly = np.pi
nz = 256
nzDNS = 500
xgrid = np.linspace(0, Lx, nx + 1)
ygrid = np.linspace(0, Ly, ny + 1)
xu = xgrid[1:]
xp = (xgrid[:-1] + xu) / 2
yv = ygrid[1:]
yp = (ygrid[:-1] + yv) / 2

dz = np.squeeze(read_bin('inputdir/dz_grid_590.dat', (nzDNS, 1)))
zp = get_zp(dz)
zp = np.concatenate(([-1], zp - 1, [1]))
_, zw = cheb(nzDNS)
zw = np.flipud(zw)

kx = np.concatenate((np.arange(21), np.arange(-20, 0)))
ky = np.arange(0, 41, 2)
kx_array = np.repeat(kx[:], len(ky), axis=0)
ky_array = np.tile(ky, len(kx))

np.savez('full590_mean.npz', channelRe=channelRe, nx=nx, ny=ny, nz=nz, nzDNS=nzDNS, dkx=dkx, dky=dky, Lx=Lx, Ly=Ly, xu=xu, xp=xp, yv=yv, yp=yp, zw=zw, zp=zp, kx_array=kx_array, ky_array=ky_array)
