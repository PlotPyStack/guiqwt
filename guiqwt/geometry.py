# -*- coding: utf-8 -*-
"""
Basic geometry functions
"""

from numpy import (matrix, array, arccos, sign, cos, sin, linalg, vdot,
                   pi, sqrt)


#===============================================================================
# Transform matrix functions
#===============================================================================

def translate(tx, ty):
    """Return translation matrix (NumPy matrix object)"""
    return matrix([[1, 0, tx],
                   [0, 1, ty],
                   [0, 0, 1 ]], float)

def scale(sx, sy):
    """Return scale matrix (NumPy matrix object)"""
    return matrix([[sx, 0,  0],
                   [0,  sy, 0],
                   [0,  0,  1]], float)

def rotate(alpha):
    """Return rotation matrix (NumPy matrix object)"""
    return matrix([[cos(alpha), -sin(alpha), 0],
                   [sin(alpha),  cos(alpha), 0],
                   [0,           0,          1]], float)

def colvector(x, y):
    """Return vector (NumPy matrix object) from coordinates"""
    # z component must be null, otherwise the vector norm would be incorrect
    return matrix([x, y, 0]).T


#===============================================================================
# Operations on vectors (from coordinates)
#===============================================================================

def vector_norm(xa, ya, xb, yb):
    """Return vector norm: (xa, xb)-->(ya, yb)"""
    return linalg.norm(array((xb-xa, yb-ya)))

def vector_projection(dv, xa, ya, xb, yb):
    """Return vector projection on *dv*: (xa, xb)-->(ya, yb)"""
    assert dv.shape == (2,)
    v_ab = array((xb-xa, yb-ya))
    u_ab = v_ab/linalg.norm(v_ab)
    return vdot(u_ab, dv)*u_ab+array((xb, yb))

def vector_rotation(theta, dx, dy):
    """Compute theta-rotation on vector *v*, returns vector coordinates"""
    return array( rotate(theta)*colvector(dx, dy) ).ravel()[:2]

def vector_angle(dx, dy):
    """Return vector angle with X-axis"""
    # sign(dy) ==  1 --> return Arccos()
    # sign(dy) ==  0 --> return  0 if sign(dx) ==  1
    # sign(dy) ==  0 --> return pi if sign(dx) == -1
    # sign(dy) == -1 --> return 2pi-Arccos()
    if dx == 0 and dy == 0:
        return 0.
    else:
        sx, sy = sign(dx), sign(dy)
        acos = arccos(dx/sqrt(dx**2+dy**2))
        return sy*(pi*(sy-1)+acos)+pi*(1-sy**2)*(1-sx)*.5
