# -*- coding: utf-8 -*-
#
# Copyright © 2009-2011 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""Logarithmic scale test for curve plotting"""

SHOW = False # Do not show test in GUI-based test launcher

from guiqwt.plot import CurveDialog
from guiqwt.builder import make

def test():
    """Test"""
    # -- Create QApplication
    import guidata
    _app = guidata.qapplication()
    # --
    import numpy as np
    x = np.linspace(-10, 10, 200)
    y = np.exp(-x)
    item = make.curve(x, y, color="b")
    
    win = CurveDialog()
    plot = win.get_plot()
    plot.add_item(item)
    plot.set_axis_scale("left", "log")
#    plot.set_axis_limits("left", 4.53999297625e-05, 22026.4657948)
    plot.do_autoscale()
    win.show()
    win.exec_()

if __name__ == "__main__":
    test()
