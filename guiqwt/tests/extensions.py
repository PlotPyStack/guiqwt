# -*- coding: utf-8 -*-
#
# Copyright © 2012 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""Cython/Fortran extensions test"""

import time

import numpy as np
import scipy.ndimage as ni

from guiqwt._ext import hist2d, hist2d_func


def test(func, args):
    t0 = time.time()
    output = func(*args)
    print "Elapsed time: %d ms" % ((time.time()-t0)*1000)
    return output

def hist2d_test(x, y, bins):
    ny_bins, nx_bins = bins, bins
    i1, j1, i2, j2 = x.min(), y.min(), x.max(), y.max()
    data = np.zeros((ny_bins, nx_bins), float, order='F')
    logscale = False
    _, nmax = hist2d(y, x, j1, j2, i1, i2, data, logscale)
    return data

def compare(func1, func2, args):
    output1 = test(func1, args)
    output2 = test(func2, args)
    return output1, output2, np.sum(np.abs(output2-output1))


if __name__ == "__main__":
    import guidata
    _app = guidata.qapplication()
    N = 150000
    m = np.array([[ 1., .2], [-.2, 3.]])
    X1 = np.random.normal(0, .3, size=(N, 2))
    X2 = np.random.normal(0, .3, size=(N, 2))
    X = np.concatenate((X1+[0, 1.], np.dot(X2, m)+[-1, -1.])) 
    args = X[:, 0], X[:, 1], 50
    output1, output2, norm = compare(hist2d_test, hist2d_test, args)
    
    from guiqwt import pyplot as plt
    plt.figure('hist2d_test')
    plt.imshow(output1, interpolation='nearest')
    plt.figure('hist2d_test2')
    plt.imshow(output2, interpolation='nearest')
    plt.show()
