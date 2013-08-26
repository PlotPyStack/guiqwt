# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""Mandelbrot algorithm"""

cimport cython
cimport numpy as np

@cython.profile(False)
cdef inline int mandel(double real, double imag, int nmax):
    cdef double z_real=0., z_imag=0.
    cdef int i
    for i in range(nmax):
        z_real, z_imag = (z_real*z_real - z_imag*z_imag + real,
                          2*z_real*z_imag + imag)
        if z_real*z_real + z_imag*z_imag > 4:
            return i
    return -1

@cython.boundscheck(False)
def mandelbrot(double x1, double y1, double x2, double y2,
               data, unsigned int nmax):
    """Compute Mandelbrot fractal"""
    cdef double dx, dy, real, imag
    cdef unsigned int row, col
    cdef unsigned int rows = data.shape[0]
    cdef unsigned int cols = data.shape[1]
    
    dx = (x2-x1)/(cols-1)
    dy = (y2-y1)/(rows-1)
    
    for col in range(cols):
        real = x1 + col*dx
        for row in range(rows):
            imag = y1 + row*dy
            data[row, col] = mandel(real, imag, nmax)
