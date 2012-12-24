# -*- coding: utf-8 -*-
#
# Copyright © 2012 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""Cython/Fortran extensions test

This test module was used to compare Cython and Fortran extensions performance 
before removing the Fortran extensions.
"""

from __future__ import print_function

import time
import numpy as np

from guiqwt._ext import hist2d, hist2d_func
from guiqwt.histogram2d import histogram2d, histogram2d_func


def test(func, args):
    t0 = time.time()
    output = func(*args)
    print("Elapsed time: %d ms" % ((time.time()-t0)*1000))
    return output

def hist2d_test(x, y, bins, logscale):
    ny_bins, nx_bins = bins, bins
    i1, j1, i2, j2 = x.min(), y.min(), x.max(), y.max()
    data = np.zeros((ny_bins, nx_bins), float, order='F')
    _, nmax = hist2d(y, x, j1, j2, i1, i2, data, logscale)
    print("nmax:", nmax)
    return data

def histogram2d_test(x, y, bins, logscale):
    ny_bins, nx_bins = bins, bins
    i1, j1, i2, j2 = x.min(), y.min(), x.max(), y.max()
    data = np.zeros((ny_bins, nx_bins), float)
    nmax = histogram2d(x, y, i1, i2, j1, j2, data, logscale)
    print("nmax:", nmax)
    return data

def hist2d_func_test(x, y, z, bins, computation):
    ny_bins, nx_bins = bins, bins
    i1, j1, i2, j2 = x.min(), y.min(), x.max(), y.max()
    data = np.zeros((ny_bins, nx_bins), float, order='F')
    tmp = np.array(data, copy=True)
    r = hist2d_func(y, x, z, j1, j2, i1, i2, tmp, data, computation)
    return data

def histogram2d_func_test(x, y, z, bins, computation):
    ny_bins, nx_bins = bins, bins
    i1, j1, i2, j2 = x.min(), y.min(), x.max(), y.max()
    data = np.zeros((ny_bins, nx_bins), float)
    tmp = np.array(data, copy=True)
    histogram2d_func(x, y, z, i1, i2, j1, j2, tmp, data, computation)
    return data

def compare(func1, func2, args):
    output1 = test(func1, args)
    output2 = test(func2, args)
    return output1, output2, np.sum(np.abs(output2-output1))

def show_comparison(func1, func2, args):
    output1, output2, norm = compare(func1, func2, args)
    from guiqwt import pyplot as plt
    for title, output in ((func1.__name__, output1),
                          (func2.__name__, output2)):
        plt.figure(title)
        plt.imshow(output, interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    import guidata
    _app = guidata.qapplication()
    N = 5000000
    m = np.array([[ 1., .2], [-.2, 3.]])
    X1 = np.random.normal(0, .3, size=(N, 2))
    X2 = np.random.normal(0, .3, size=(N, 2))
    X = np.concatenate((X1+[0, 1.], np.dot(X2, m)+[-1, -1.])) 

    args = X[:, 0], X[:, 1], X[:, 0], 50, 2
    show_comparison(hist2d_func_test, histogram2d_func_test, args)
    args = X[:, 0], X[:, 1], X[:, 0], 50, 4
    show_comparison(hist2d_func_test, histogram2d_func_test, args)

    args = X[:, 0], X[:, 1], 50, False
    show_comparison(hist2d_test, histogram2d_test, args)
    args = X[:, 0], X[:, 1], 50, True
    show_comparison(hist2d_test, histogram2d_test, args)
