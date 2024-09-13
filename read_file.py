import netCDF4
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d

# read the dat files containing velocities
def read_bin(filename, dim):
    with open(filename, 'rb') as fid:
        A = np.fromfile(fid, dtype=np.float64, count=np.prod(dim))
        A = np.reshape(A, tuple(dim),order='F') 
    return A

# the wall-normal coordinate of the DNS grid
def get_zp(dz):
    zp = (np.cumsum(np.concatenate(([0], dz[:-1]))) + np.cumsum(dz)) / 2
    return zp

# linearly interpolate the velocities on the staggered grid onto the collocated grid
def get_intepolated_uvw(u_old,v_old,w_old,xu,xp,yv,yp,zp,zc,zw):
    u_new = interp1d(zp,u_old,kind='cubic',axis=0,fill_value="extrapolate")(zc)
    u_new = u_new[1:-1,:,:]
    u_new = interp1d(xu,u_new,kind='cubic',axis=2,fill_value="extrapolate")(xp)
    
    v_new = interp1d(zp,v_old,kind='cubic',axis=0,fill_value="extrapolate")(zc)
    v_new = v_new[1:-1,:,:]
    v_new = interp1d(yv,v_new,kind='cubic',axis=1,fill_value="extrapolate")(yp)
    
    w_new = interp1d(zw,w_old,kind='cubic',axis=0,fill_value="extrapolate")(zc)
    w_new = w_new[1:-1,:,:]
    
    return u_new, v_new, w_new

def get_uvw(xu,xp,yv,yp,zp,zc,zw,nzDNS,ny,nx,loadname_u,loadname_v,loadname_w):
    u = np.zeros((nzDNS+2,ny,nx))
    v = np.zeros((nzDNS+2,ny,nx))
    w = np.zeros((nzDNS+1,ny,nx))
    
    u[1:-1,:,:] = read_bin(loadname_u, np.array([nzDNS,ny,nx]))
    v[1:-1,:,:] = read_bin(loadname_v, np.array([nzDNS,ny,nx]))
    w[1:,:,:]   = read_bin(loadname_w, np.array([nzDNS,ny,nx]))
    
    u,v,w = get_intepolated_uvw(u,v,w,xu,xp,yv,yp,zp,zc,zw)
    
    return u,v,w


def get_xyzuvw_channelflow(filename):
    file2read = netCDF4.Dataset(filename,'r')
    
    X_temp = file2read.variables['X']
    X = np.asarray(X_temp[:])
    del X_temp
    
    Y_temp = file2read.variables['Z']
    Y = np.asarray(Y_temp[:])
    del Y_temp
    
    Z_temp = file2read.variables['Y']
    Z = np.asarray(Z_temp[:])
    del Z_temp
    
    u_temp = file2read.variables['Velocity_X']
    u = ma.getdata(u_temp)
    u = np.asarray(u)
    u = np.transpose(u, (1, 0, 2))
    del u_temp
    
    v_temp = file2read.variables['Velocity_Z']
    v = ma.getdata(v_temp)
    v = np.asarray(v)
    v = np.transpose(v, (1, 0, 2))
    del v_temp
    
    w_temp = file2read.variables['Velocity_Y']
    w = ma.getdata(w_temp)
    w = np.asarray(w)
    w = np.transpose(w, (1, 0, 2))
    del w_temp
    
    return X,Y,Z,u,v,w







