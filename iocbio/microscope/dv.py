"""Provides dv function.

Module content
--------------
"""

__all__ = ['dv']

from numpy import *

def dx(f,h):
    """
    f - 3D array
    h = [hx, hy, hz]
    """
    shape = list(f.shape)
    shape[0] += 1
    result = zeros(shape)
    result[-1] = f[-1]
    result[0] = -f[0]
    result[:-1] += f
    result[1:] -= f
    result /= h[0]
    #return dx+       , dx-
    return  result[1:], result[:-1]

def dy(f,h):
    shape = list(f.shape)
    shape[1] += 1
    result = zeros(shape)
    result[:,-1] = f[:,-1]
    result[:,0] = -f[:,0]
    result[:,:-1] += f
    result[:,1:] -= f
    result /= h[1]
    #return dy+       , dy-
    return  result[:,1:], result[:,:-1]

def dz(f,h):
    shape = list(f.shape)
    shape[2] += 1
    result = zeros(shape)
    result[:,:,-1] = f[:,:,-1]
    result[:,:,0] = -f[:,:,0]
    result[:,:,:-1] += f
    result[:,:,1:] -= f
    result /= h[2]
    #return dz+       , dz-
    return  result[:,:,1:], result[:,:,:-1]

def m(a, b):
    return 0.5*(sign(a) + sign(b)) * minimum(absolute(a), absolute(b))

def dv(f, h):
    """Computes ``div(grad(f)/|grad(f)|)``.
    
    Parameters
    ----------
    f : :numpy:`ndarray`
    h : 3-tuple
    """
    fxp, fxm = dx(f, h)
    fyp, fym = dy(f, h)
    fzp, fzm = dz(f, h)
    mx = m(fxp, fxm)
    my = m(fyp, fym)
    mz = m(fzp, fzm)
    mx2 = mx ** 2
    my2 = my ** 2
    mz2 = mz ** 2
    fx2 = fxp**2
    fy2 = fyp**2
    fz2 = fzp**2
    sx = sqrt(fx2 + my2 + mz2)
    sy = sqrt(fy2 + mx2 + mz2)
    sz = sqrt(fz2 + my2 + mx2)
    fsx = fxp/sx
    fsy = fyp/sy
    fsz = fzp/sz
    fx = dx(where(sx==0, 0, fsx), h)[1]
    fy = dy(where(sy==0, 0, fsy), h)[1]
    fz = dz(where(sz==0, 0, fsz), h)[1]
    return fx + fy + fz
