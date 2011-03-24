# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Ludovic Aubry
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""ImageDialog / Pcolor test"""

SHOW = True # Show test in GUI-based test launcher

import numpy as np

from guiqwt.plot import ImageDialog
from guiqwt.builder import make

def imshow( items ):
    win = ImageDialog(edit=False, toolbar=True, options={"yreverse": False},
                      wintitle="Pcolor test")
    plot = win.get_plot()
    for item in items:
        plot.add_item(item)
    win.show()
    win.exec_()

def compute_quads(N=100):
    r = np.linspace(1., 16, N)
    th= np.linspace(0., np.pi, N)

    R,TH=np.meshgrid(r,th)
    X = R*np.cos(TH)
    Y = R*np.sin(TH)
    Z = 4*TH+R
    item = make.pcolor(X,Y,Z)
    return [item]

def compute_quads2(N=4):
    x = np.linspace(0, N-1, N)
    X, Y = np.meshgrid(x, x)
    Z = X**2+Y**2
    Z = Z.reshape((8, 2))
    item = make.pcolor(Z) # checks if X, Y are properly generated in make.pcolor
    return [item]

def compute_quads3():
    pi = np.pi
    cos = np.cos
    sin = np.sin
    items = []
    for i,t in enumerate( np.linspace(0,2*pi,16) ): 
        X = np.array( [[    0.0, cos(t)],
                       [-sin(t), cos(t)-sin(t)]] )
        Y = np.array( [[   0.0, sin(t)],
                       [cos(t), sin(t)+cos(t)]] )
        Z = np.array([[1.,2.],[3.,4.]])
        item = make.pcolor(X-16+2*i,Y-3,Z)
        items.append(item)
    return items

def test():
    """Test"""
    # -- Create QApplication
    import guidata
    guidata.qapplication()
    # --    
    imshow(compute_quads()+compute_quads3())

if __name__ == "__main__":
    test()