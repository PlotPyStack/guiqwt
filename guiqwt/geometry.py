# -*- coding: utf-8 -*-
"""
Basic geometry functions
"""

import numpy as np

def translate(tx, ty):
    """Return translation matrix (NumPy matrix object)"""
    return np.matrix([[1, 0, tx],
                      [0, 1, ty],
                      [0, 0, 1 ]], float)

def scale(sx, sy):
    """Return scale matrix (NumPy matrix object)"""
    return np.matrix([[sx, 0,  0],
                      [0,  sy, 0],
                      [0,  0,  1]], float)

def rotate(alpha):
    """Return rotation matrix (NumPy matrix object)"""
    return np.matrix([[np.cos(alpha), -np.sin(alpha), 0],
                      [np.sin(alpha),  np.cos(alpha), 0],
                      [0,              0,             1]], float)

def vector(x,y):
    """Return vector (NumPy matrix object) from coordinates"""
    return np.matrix([x, y, 1]).T

