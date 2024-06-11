import numpy as np
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
    u_new = interp1d(zp,u_old,kind='linear',axis=0,fill_value="extrapolate")(zc)
    u_new = u_new[1:-1,:,:]
    u_new = interp1d(xu,u_new,kind='linear',axis=2,fill_value="extrapolate")(xp)
    
    v_new = interp1d(zp,v_old,kind='linear',axis=0,fill_value="extrapolate")(zc)
    v_new = v_new[1:-1,:,:]
    v_new = interp1d(yv,v_new,kind='linear',axis=1,fill_value="extrapolate")(yp)
    
    w_new = interp1d(zw,w_old,kind='linear',axis=0,fill_value="extrapolate")(zc)
    w_new = w_new[1:-1,:,:]
    
    return u_new, v_new, w_new







