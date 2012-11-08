# -*- coding: utf-8 -*-
#
# Copyright © 2012 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""2D-Histogram algorithm"""

cimport cython
cimport numpy as np

import numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef inline double double_max(double a, double b): return a if a >= b else b

@cython.profile(False)
@cython.boundscheck(False)
def histogram2d(np.ndarray[double, ndim=1] X, np.ndarray[double, ndim=1] Y,
                double i0, double i1, double j0, double j1,
                np.ndarray[double, ndim=2] data, logscale):
    """Compute 2-D Histogram from data X, Y"""
    cdef double cx, cy, nmax, ix, iy
    cdef unsigned int i
    cdef unsigned int n = X.shape[0]
    cdef unsigned int nx = data.shape[1]
    cdef unsigned int ny = data.shape[0]
    
    cx = nx/(i1-i0)
    cy = ny/(j1-j0)
    
    for i in range(n):
        #  Centered bins => - .5
        ix = (X[i]-i0)*cx - .5
        iy = (Y[i]-j0)*cy - .5
        if ix >= 0 and ix <= nx-1 and iy >= 0 and iy <= ny-1:
            data[<int> iy, <int> ix] += 1

    nmax = 0.
    if logscale:
        for j in range(ny):
            for i in range(nx):
                data[j, i] = np.log(1+data[j, i])
                nmax = double_max(nmax, data[j, i])
    else:
        for j in range(ny):
            for i in range(nx):
                nmax = double_max(nmax, data[j, i])
    return nmax
